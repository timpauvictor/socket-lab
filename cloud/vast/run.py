"""Run a historical Vast job with artifact finalization and a stop-only guard.

Execute on the instance with /venv/main/bin/python. No instance is bought,
started or destroyed by this command. --dry-run performs no writes or calls.
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

if __package__:
    from .lifecycle import container_credentials, markers, stop_self
    from .recipes import RECIPES, Job, compile_jobs, job_spec
else:
    from lifecycle import container_credentials, markers, stop_self
    from recipes import RECIPES, Job, compile_jobs, job_spec

LIFECYCLE = Path(__file__).with_name("lifecycle.py").resolve()


def terminate(process: subprocess.Popen) -> None:
    """Stop only a process/group created by this runner."""
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()
    except ProcessLookupError:
        process.wait()


def finish_command(job: Job, python: str, returncode: int) -> list[str]:
    command = [python, str(LIFECYCLE), "finish", job.tag, str(job.output), job.repo,
               str(returncode), str(job.log)]
    if job.keep_running:
        command += ["--keep-running"]
    return command


def run_job(job: Job, python: str, *, finalize_success: bool = True) -> int:
    container_credentials()
    job.log.parent.mkdir(parents=True, exist_ok=True)
    for marker in markers(job.log, job.tag):
        marker.unlink(missing_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    guard_command = [python, str(LIFECYCLE), "watchdog", str(job.log), job.completion, job.tag]
    guard = None
    trainer = None
    returncode = 1
    started = time.monotonic()
    try:
        with job.log.open("w", encoding="utf-8") as log, job.log.with_name(f"watchdog-{job.tag}.log").open("a", encoding="utf-8") as guard_log:
            guard = subprocess.Popen(guard_command, stdout=guard_log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, env=env, start_new_session=True)
            trainer = subprocess.Popen(job.command, cwd=job.log.parent.parent / "scripts",
                                       stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       env=env, start_new_session=True)
            while trainer.poll() is None:
                if guard.poll() is not None:
                    raise RuntimeError("Watchdog exited before training finished")
                try:
                    trainer.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            if guard.poll() is not None:
                raise RuntimeError("Watchdog exited before finalization")
            returncode = trainer.returncode
    except KeyboardInterrupt:
        print("Interrupted; cancelling owned training, preserving logs and stopping the instance.", flush=True)
        returncode = 130
    except (OSError, RuntimeError) as exc:
        returncode = 1
        print(f"Training launch/guard failure ({type(exc).__name__}); attempting finalization.", flush=True)
        if guard is None or guard.poll() is not None:
            stop_self()  # Do not rely on an absent watchdog while uploading.
    finally:
        if trainer is not None:
            try:
                terminate(trainer)
            except OSError:
                returncode = 1
                print("Could not cancel the owned trainer; finalization will request an instance stop.", flush=True)
    print(f"{job.tag}: {time.monotonic() - started:.0f}s | training rc={returncode} | log={job.log}", flush=True)
    # The compilation comparison historically leaves successful runs on disk.
    # A failure still takes the log-upload/stop path.
    if returncode == 0 and not finalize_success:
        if guard is not None:
            terminate(guard)
        return 0
    try:
        finalized = subprocess.run(finish_command(job, python, returncode), env=env).returncode
    except (OSError, KeyboardInterrupt):
        print("Finalizer could not finish; requesting stop directly.", flush=True)
        stop_self()
        finalized = 1
    if finalized == 0 and guard is not None:
        terminate(guard)
    elif guard is not None:
        print("Finalization failed; independent watchdog remains active until the instance stops.", flush=True)
    if returncode:
        return 128 - returncode if returncode < 0 else returncode
    return finalized


def interrupt_job(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=[*RECIPES, "compile-smoke"])
    parser.add_argument("--workspace", type=Path, default=Path("/workspace/bh"))
    parser.add_argument("--python", default="/venv/main/bin/python")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                        help="override the historical recipe's compile setting")
    parser.add_argument("--resume-from")
    parser.add_argument("--resume-pair")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    admitted = False
    try:
        if args.job == "compile-smoke":
            if args.compile is not None or args.resume_from or args.resume_pair:
                parser.error("compile-smoke has fixed plain/compiled arms and no resume")
            jobs = compile_jobs(args.workspace.resolve(), args.python)
        else:
            jobs = [job_spec(args.job, args.workspace.resolve(), args.python,
                             compile_model=args.compile, resume_from=args.resume_from, resume_pair=args.resume_pair)]
        if args.dry_run:
            for job in jobs:
                print(json.dumps(job.command))
                print(f"Log: {job.log}; guarded by an independent watchdog.")
                print("Successful compile checks remain local; failures upload logs and stop." if args.job == "compile-smoke"
                      else json.dumps(finish_command(job, args.python, 0)))
            return 0
        if not shutil.which(args.python):
            raise ValueError("Python executable not found; use the instance's training environment")
        for job in jobs:
            for path in (Path(job.command[1]), Path(job.command[1]).with_name("translation_data.py"), LIFECYCLE):
                if not path.is_file():
                    raise ValueError(f"Required uploaded script is missing: {path}")
        container_credentials()
        admitted = True
        previous = signal.signal(signal.SIGTERM, interrupt_job)
        try:
            for job in jobs:
                code = run_job(job, args.python, finalize_success=args.job != "compile-smoke")
                if args.job == "compile-smoke":
                    losses = re.findall(r"'train_loss':\s*([0-9.]+)", job.log.read_text(encoding="utf-8", errors="replace"))
                    print(f"{job.tag}: train_loss={losses[-1] if losses else 'not found'}")
                if code:
                    return code
        finally:
            signal.signal(signal.SIGTERM, previous)
        if args.job == "compile-smoke":
            print("Both arms ran. Compare total time and loss; compilation is not automatically faster. Instance remains running.")
        return 0
    except KeyboardInterrupt:
        if admitted:
            stop_self()
        return 130
    except (OSError, ValueError) as exc:
        if admitted:
            stop_self()
        print(f"Job error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
