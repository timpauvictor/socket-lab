from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from tools.processes import resolve_executable, run_logged

ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a Modal training job.")
    parser.add_argument("--job", required=True, choices=("box", "task", "monolith", "eos-tune", "smoke"))
    parser.add_argument("--phase", required=True, choices=("smoke", "full"))
    parser.add_argument("--lang", default="ro", choices=("ro", "en", "fr", "de", "all"))
    parser.add_argument("--modal-exe", default="modal")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--log-file", type=Path, help="Fresh console log; relative paths use --repo-root.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def command_for(args: argparse.Namespace) -> tuple[list[str], str]:
    if args.lang == "all" and args.job != "box":
        raise ValueError("--lang all is only valid with --job box")
    if args.job == "smoke":
        if args.phase != "smoke":
            raise ValueError("The standalone smoke job only supports --phase smoke")
        target, log_name = "cloud/modal/train_smoke.py::app.smoke", "train-smoke"
    else:
        module = args.job.replace("-", "_")
        phase = args.phase + ("_all" if args.lang == "all" else "")
        target = f"cloud/modal/train_{module}.py::app.{phase}"
        log_name = f"train-{args.job}-{args.phase}"
        if args.job == "box":
            log_name = f"train-box-{args.lang}-{args.phase}"
    executable = args.modal_exe if args.dry_run else resolve_executable(args.modal_exe)
    command = [executable, "run", target]
    if args.job == "box" and args.lang != "all":
        command += ["--lang", args.lang]
    return command, str(args.log_file) if args.log_file is not None else f"logs/{log_name}-console.log"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        command, log = command_for(args)
        if args.dry_run:
            print(shlex.join(command))
            print(f"log: {log}")
            return 0
        print(f"Modal {args.job}/{args.phase} uses paid GPU time; smoke checks must pass before full runs.", flush=True)
        env = os.environ.copy()
        env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        return run_logged(command, cwd=args.repo_root, env=env, log_path=args.repo_root / log)
    except (OSError, ValueError) as exc:
        print(f"run_train: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
