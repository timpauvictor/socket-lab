from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from tools.processes import resolve_executable, run_foreground

ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local llama.cpp server in the foreground.")
    parser.add_argument("--server-exe", default=os.environ.get("LLAMA_SERVER") or "llama-server")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "qwen3-0.6b-base-q8_0.gguf")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model-alias", default="socket-lab")
    parser.add_argument("--lora", action="append", default=[])
    parser.add_argument("--context-size", type=int, default=8192)
    parser.add_argument("--host-address", default="127.0.0.1")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def command_for(args: argparse.Namespace) -> list[str]:
    if not args.model.is_file():
        raise ValueError(f"Model file not found: {args.model}")
    adapters = []
    for adapter in args.lora:
        if not adapter:
            continue
        path = Path(adapter)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise ValueError(f"Adapter file not found: {path}")
        adapters.append(str(path))
    executable = args.server_exe if args.dry_run else resolve_executable(args.server_exe)
    command = [executable, "-m", str(args.model), "--alias", args.model_alias,
               "-c", str(args.context_size), "-ngl", "99", "--cache-ram", "0",
               "--ctx-checkpoints", "4", "--host", args.host_address, "--port", str(args.port)]
    if adapters:
        command += ["--lora", ",".join(adapters)]
    return command


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        command = command_for(args)
        if args.dry_run:
            print(shlex.join(command))
            return 0
        env = os.environ.copy()
        env["DISABLE_RTSS_LAYER"] = "1"
        return run_foreground(command, cwd=Path.cwd(), env=env)
    except (OSError, ValueError) as exc:
        print(f"serve: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
