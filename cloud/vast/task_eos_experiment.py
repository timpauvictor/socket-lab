#!/usr/bin/env python3
"""One supervised controlled task-EOS experiment. Dry-run is strictly read-only.

Run only on an explicitly approved Vast instance. Never provisions or destroys.
"""
import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloud.modal.artifacts import write_manifest
from cloud.vast.lifecycle import container_credentials, stop_self, watchdog as lifecycle_watchdog
from cloud.vast.run import interrupt_job, terminate
from cloud.vast.task_eos_artifacts import (DIRECTIONS, MODEL, PAIRS, check_frozen,
                                          check_revision, input_names, SOURCES)

ROOT = Path(__file__).resolve().parents[2]


def stage_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """Construct argv only: no filesystem/network/process activity."""
    repo, run = args.repo, args.output
    inputs = run / "bundle/inputs"
    bundle = run / "bundle"
    python = args.python
    commands = [("freeze", [python, str(repo / "cloud/vast/task_eos_artifacts.py"), "freeze",
        "--repo", str(repo), "--output", str(inputs), "--model-revision", args.model_revision,
        "--dataset-revision", args.dataset_revision, "--provenance", str(args.provenance)])]
    for phase, steps in (("smoke", 9), ("full", 1000)):
        if phase == "full" and not args.full:
            continue
        for arm in ("B", "C"):
            command = [python, str(repo / "training/train_task.py"), "--model", MODEL,
                "--revision", args.model_revision, "--output", str(bundle / phase / arm),
                "--tune_from", str(inputs / "adapters/task-full"), "--total_steps", str(steps),
                "--pairs", ",".join(PAIRS), "--rank", "32", "--lr", "5e-5", "--seed", "42",
                "--batch_size", "8", "--grad_accum", "4", "--prepared_data", str(inputs / "data"),
                "--paired_control"]
            for lang in ("ro", "en", "fr", "de"):
                command += [f"--box_{lang}", str(inputs / f"adapters/box-{lang}")]
            if arm == "C":
                command += ["--eos_stop"]
            if phase == "smoke":
                command += ["--verify_updates"]
            commands.append((f"{phase}-train-{arm}", command))
        for arm in "ABCD":
            adapter = inputs / ("adapters/eos-tune-001" if arm == "D" else "adapters/task-full")
            if arm in ("B", "C"):
                adapter = bundle / phase / arm
            for direction in DIRECTIONS:
                src, tgt = direction.split("-")
                command = [python, str(repo / "evaluation/eval_task_eos.py"), "--revision", args.model_revision,
                    "--inputs", str(inputs), "--adapter", str(adapter), "--arm", arm,
                    "--src", src, "--tgt", tgt, "--pairs", str(inputs / f"logs/eval-{direction}-devtest.jsonl"),
                    "--output", str(bundle / phase / f"eval-{arm}-{direction}.jsonl"), "--run-id", args.run_id]
                if phase == "smoke":
                    command += ["--limit", "2"]
                commands.append((f"{phase}-eval-{arm}-{direction}", command))
        commands.append((f"{phase}-report", [python, str(repo / "cloud/vast/task_eos_experiment.py"),
                                            "report", str(bundle / phase)]))
    return commands


def report(directory: Path) -> None:
    from evaluation.eval_task_eos import summarize
    rows = [summarize(directory / f"eval-{arm}-{direction}.jsonl") for arm in "ABCD" for direction in DIRECTIONS]
    write_manifest(directory / "scores.json", rows)
    lines = ["# Controlled EOS experiment", "", "| Arm | Direction | chrF | Repetition | EOS | Cap | Success/expected |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        rates = [f"{row[key]:.4f}" if row[key] is not None else "incomplete"
                 for key in ("repetition_fraction", "eos_stop_rate", "token_cap_rate")]
        lines.append(f"| {row['arm']} | {row['direction']} | {row['chrf']} | {' | '.join(rates)} | "
                     f"{row['successful']}/{row['expected']} |")
    lines += ["", "chrF differences (C-B: EOS; B-A: extra training; C-A: practical tune-up):"]
    for direction in DIRECTIONS:
        values = {r["arm"]: r["chrf"] for r in rows if r["direction"] == direction}
        if all(v is not None for v in values.values()):
            lines.append(f"- {direction}: C-B={values['C']-values['B']:+.2f}; "
                         f"B-A={values['B']-values['A']:+.2f}; C-A={values['C']-values['A']:+.2f}")
    lines += ["", "D is a fresh reference, not an isolated architecture comparison. One seed is not significance.",
              "French is trained but not evaluated. Check every direction: mixed/negative outcomes are valid.",
              "Incomplete outputs are not valid final comparisons. JSON retains signatures, lengths and failure counts."]
    with (directory / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    if not all(row["complete"] for row in rows):
        raise RuntimeError("Incomplete evaluation matrix")


def watchdog(progress: Path, deadline: float) -> int:
    """Independent wall-clock stop guard; survives supervisor death/finalizer hang."""
    return lifecycle_watchdog(progress, "EOS_ALL_STAGES_FINISHED", "task-eos", deadline=deadline, interval=30)


def run_stage(name: str, command: list[str], run: Path, guard: subprocess.Popen,
              deadline: float, progress: Path) -> None:
    process = None
    try:
        with (run / f"{name}.stdout.log").open("x", encoding="utf-8") as out, (run / f"{name}.stderr.log").open("x", encoding="utf-8") as err:
            process = subprocess.Popen(command, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1"))
            while process.poll() is None:
                if guard.poll() is not None:
                    raise RuntimeError("Independent watchdog exited unexpectedly")
                if time.time() >= deadline:
                    raise RuntimeError("Approved execution time exhausted")
                with progress.open("a", encoding="utf-8") as log:
                    log.write(f"{time.time():.0f} {name} active\n")
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                raise RuntimeError(f"Stage {name} exited {process.returncode}; logs retained")
            if guard.poll() is not None:
                raise RuntimeError("Watchdog exited at stage boundary")
    finally:
        if process is not None:
            terminate(process)


def validate_live(args: argparse.Namespace) -> None:
    check_revision(args.model_revision)
    check_revision(args.dataset_revision)
    if not args.approved_live:
        raise ValueError("Explicit --approved-live required; read prerequisites first")
    if any(not math.isfinite(v) or v <= 0 for v in (args.max_seconds, args.max_cost, args.hourly_rate)):
        raise ValueError("Positive finite max-seconds, max-cost and hourly-rate required")
    if args.max_seconds < 120:
        raise ValueError("Budget must leave at least 60 seconds for failure finalization")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("Unique safe run ID required")
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", args.destination):
        raise ValueError("New private owner/repository destination required")
    if args.destination.rsplit("/", 1)[1] != args.run_id:
        raise ValueError("Destination repository name must equal the unique run ID")
    if args.output.exists():
        raise ValueError("Fresh experiment output required; no automatic training resume")
    if not args.provenance.is_file():
        raise ValueError("Inspected artifact provenance file required")
    for name in (*SOURCES, *input_names()):
        if not (args.repo / name).is_file():
            raise ValueError(f"Missing staged input/source: {name}")
    container_credentials()


def run_experiment(args: argparse.Namespace) -> int:
    validate_live(args)
    started = time.time()
    seconds = min(args.max_seconds, args.max_cost / args.hourly_rate * 3600)
    if seconds < 120:
        raise ValueError("Spend ceiling leaves insufficient finalization time")
    deadline = started + seconds
    work_deadline = deadline - 60
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = args.output / "bundle"
    bundle.mkdir()
    logs = bundle / "stage-logs"
    logs.mkdir()
    progress = args.output / "progress.log"
    progress.touch()
    guard = None
    succeeded = False
    uploaded = False
    stopped = False
    failure = None
    completed = []
    previous = signal.signal(signal.SIGTERM, interrupt_job)
    try:
        write_manifest(bundle / "request.json", {"run_id": args.run_id,
            "model_revision": args.model_revision, "dataset_revision": args.dataset_revision,
            "full": args.full, "max_seconds": args.max_seconds, "max_cost": args.max_cost,
            "hourly_rate": args.hourly_rate, "effective_deadline_unix": deadline,
            "cost_scope": "GPU rate estimate only; actual billing/storage requires dashboard evidence",
            "commands": stage_commands(args)})
        with (args.output / "watchdog.log").open("x", encoding="utf-8") as guard_log:
            guard = subprocess.Popen([args.python, str(args.repo / "cloud/vast/task_eos_experiment.py"),
                "watchdog", str(progress), str(deadline)], stdout=guard_log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        for name, command in stage_commands(args):
            if name != "freeze":
                check_frozen(bundle / "inputs", args.repo)
            run_stage(name, command, logs, guard, work_deadline, progress)
            completed.append(name)
            check_frozen(bundle / "inputs", args.repo)
        succeeded = True
    except (Exception, KeyboardInterrupt) as exc:
        failure = type(exc).__name__
        print(f"Experiment incomplete ({failure}); preserving evidence and stopping.", flush=True)
    finally:
        try:
            write_manifest(bundle / "completion.json", {"status": "complete" if succeeded and args.full else
                "smoke_complete" if succeeded else "incomplete", "completed_stages": completed,
                "failure_type": failure, "elapsed_seconds": time.time() - started,
                "estimated_gpu_cost_so_far": (time.time() - started) * args.hourly_rate / 3600,
                "actual_spend": "Unverified; user dashboard receipt required",
                "upload_verified": False, "stop_acknowledged": False,
                "note": "Pre-finalization snapshot; finalization/stop receipts remain beside bundle"})
            if guard is None or guard.poll() is not None:
                # Never rely on an absent guard during network finalization.
                stopped = stop_self()
            else:
                import shutil
                shutil.copyfile(progress, bundle / "progress.log")
                shutil.copyfile(args.output / "watchdog.log", bundle / "watchdog.log")
                command = [args.python, str(args.repo / "cloud/vast/task_eos_artifacts.py"),
                           "finalize", str(bundle), args.destination]
                run_stage("finalize", command, args.output, guard, deadline, progress)
                uploaded = True
        except (Exception, KeyboardInterrupt) as exc:
            print(f"Bundle finalization failed ({type(exc).__name__}); ALL disk data retained.", flush=True)
        finally:
            try:
                stopped = stop_self() or stopped
                if stopped and guard is not None:
                    terminate(guard)
                write_manifest(args.output / "closeout.json", {"upload_verified": uploaded,
                    "stop_acknowledged": stopped, "elapsed_seconds": time.time() - started,
                    "dashboard_billing_confirmation": "User required; storage may still bill"})
            finally:
                signal.signal(signal.SIGTERM, previous)
    return 0 if succeeded and uploaded and stopped else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--repo", type=Path, default=ROOT)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--destination", required=True)
    run.add_argument("--model-revision", default="RESOLVE_BEFORE_LIVE")
    run.add_argument("--dataset-revision", default="RESOLVE_BEFORE_LIVE")
    run.add_argument("--provenance", type=Path, default=Path("provenance.json"))
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--max-seconds", type=float, default=0)
    run.add_argument("--max-cost", type=float, default=0)
    run.add_argument("--hourly-rate", type=float, default=0)
    run.add_argument("--approved-live", action="store_true")
    run.add_argument("--full", action="store_true", help="After smoke, continue to both full arms and final evaluation")
    run.add_argument("--dry-run", action="store_true")
    guard = sub.add_parser("watchdog")
    guard.add_argument("progress", type=Path)
    guard.add_argument("deadline", type=float)
    table = sub.add_parser("report")
    table.add_argument("directory", type=Path)
    staging = sub.add_parser("stage-list", help="Print the explicit archive allowlist; no writes or connections")
    staging.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.command == "stage-list":
        names = set(SOURCES) | set(input_names())
        for adapter in ("box-ro", "box-en", "box-fr", "box-de", "task-full", "eos-tune-001"):
            for name in ("train.log", "run-info.json", "manifest.json", "_COMPLETE.json"):
                path = f"adapters/{adapter}/{name}"
                if (args.repo / path).is_file():
                    names.add(path)
        print("\n".join(sorted(names)))
        return 0
    if args.command == "watchdog":
        return watchdog(args.progress, args.deadline)
    if args.command == "report":
        report(args.directory)
        return 0
    if args.dry_run:
        for name, command in stage_commands(args):
            print(json.dumps({"stage": name, "argv": command}))
        print("Finalize: seal all bundle files, upload to NEW PRIVATE " + args.destination + ", download/hash-verify, stop instance.")
        print("Independent watchdog covers preparation through finalization. Live approval, provenance, revisions and ceilings still required.")
        return 0
    try:
        return run_experiment(args)
    except (OSError, ValueError) as exc:
        print(f"Experiment admission failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
