from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from tools.processes import resolve_executable, run_foreground

ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or score the local FLORES evaluation.")
    parser.add_argument("--src", default=None)
    parser.add_argument("--tgt", default=None)
    parser.add_argument("--model", default="socket-lab")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--split", default="devtest")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--base-url", default="http://localhost:8081")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--uv-exe", default="uv")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def command_for(args: argparse.Namespace) -> list[str]:
    script = args.repo_root / "evaluation" / "eval_llama.py"
    if not script.is_file():
        raise ValueError(f"Eval script not found: {script}")
    if args.score_only:
        if not args.output:
            raise ValueError("--score-only requires --output")
    elif not args.run_id.strip():
        raise ValueError("Inference requires --run-id to identify model weights, adapters, and backend.")
    executable = args.uv_exe if args.dry_run else resolve_executable(args.uv_exe)
    cli = [str(script), "--model", args.model, "--split", args.split,
           "--max_new_tokens", str(args.max_new_tokens), "--limit", str(args.limit),
           "--request_timeout", str(args.request_timeout), "--base_url", args.base_url]
    if args.score_only:
        if args.src is not None:
            cli += ["--src", args.src]
        if args.tgt is not None:
            cli += ["--tgt", args.tgt]
    else:
        cli += ["--src", args.src or "ro", "--tgt", args.tgt or "en", "--run_id", args.run_id]
    if args.output:
        cli += ["--output", args.output]
    if args.overwrite:
        cli.append("--overwrite")
    if args.score_only:
        cli.append("--score_only")
    deps = ["run", "--no-project", "--with", "sacrebleu==2.6.0"]
    if not args.score_only:
        deps += ["--with", "datasets", "--with", "requests==2.34.2"]
    return [executable, *deps, "--", "python", *cli]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        command = command_for(args)
        if args.dry_run:
            print(shlex.join(command))
            return 0
        return run_foreground(command, cwd=Path.cwd(), env=os.environ.copy())
    except (OSError, ValueError) as exc:
        print(f"run_eval: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
