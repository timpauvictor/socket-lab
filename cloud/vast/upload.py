"""Upload the existing Vast training package with explicit ssh/scp arguments."""
import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = ("ro", "en", "fr", "de")


def upload_commands(repo: Path, host: str, port: int, checkpoint: Path) -> tuple[list[list[str]], list[Path]]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:-]*", host) or not 1 <= port <= 65535:
        raise ValueError("provide a valid SSH host and port")
    address = f"[{host}]" if ":" in host else host
    remote = f"root@{address}"
    training = [repo / "training" / name for name in
                ("train_lora.py", "train_box.py", "train_task.py", "translation_data.py")]
    operations = [repo / "cloud/vast" / name for name in
                  ("run.py", "recipes.py", "lifecycle.py", "provision.py")]
    directories = ["/workspace/bh/scripts", "/workspace/bh/logs"]
    directories += [f"/workspace/bh/boxes/box-{lang}" for lang in LANGUAGES]
    # SSH runs one fixed remote executable; no shell control flow or user code.
    mkdir = "mkdir -p " + " ".join(shlex.quote(path) for path in directories)
    commands = [["ssh", "-p", str(port), remote, mkdir],
                ["scp", "-P", str(port), *map(str, training), f"{remote}:/workspace/bh/scripts/"],
                ["scp", "-P", str(port), *map(str, operations), f"{remote}:/workspace/bh/"],
                ["scp", "-P", str(port), "-r", str(checkpoint), f"{remote}:/workspace/bh/ckpt-9000"]]
    sources = [*training, *operations, checkpoint]
    for lang in LANGUAGES:
        files = [repo / "adapters" / f"box-{lang}" / name
                 for name in ("adapter_config.json", "adapter_model.safetensors")]
        sources.extend(files)
        commands.append(["scp", "-P", str(port), *map(str, files), f"{remote}:/workspace/bh/boxes/box-{lang}/"])
    return commands, sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("SSH_HOST", ""))
    parser.add_argument("--port", type=int, default=os.environ.get("SSH_PORT", "0"))
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        repo = args.repo_root.resolve()
        checkpoint = (args.checkpoint or repo / "vast/staging/ckpt-9000").resolve()
        commands, sources = upload_commands(repo, args.host, args.port, checkpoint)
        if args.dry_run:
            for command in commands:
                print(json.dumps(command))
            return 0
        for source in sources:
            if not source.exists():
                raise ValueError(f"Upload source is missing: {source}")
        for command in commands:
            result = subprocess.run(command)
            if result.returncode:
                return result.returncode
        print("Upload complete. On the instance, use /venv/main/bin/python provision.py; see cloud/vast/README.md.")
        return 0
    except (OSError, ValueError) as exc:
        print(f"Upload error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
