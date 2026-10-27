"""Provision a Vast training environment using its existing Python interpreter.

Run with /venv/main/bin/python. Torch is supplied by the image, not installed
here. --dry-run performs no installs, downloads, authentication or writes.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGES = ("transformers>=4.57,<5", "peft==0.21.2", "datasets==5.0.1",
            "accelerate==1.15.0", "huggingface_hub")
BASE_MODEL = "Qwen/Qwen3-0.6B-Base"


def provision(workspace: Path) -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise ValueError("HF_TOKEN is empty; set it privately in your SSH session")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
    subprocess.run(["uv", "pip", "install", "--quiet", "--python", sys.executable, *PACKAGES], check=True)
    import transformers
    import peft
    import triton
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_TOKEN_PATH

    print("transformers", transformers.__version__, "peft", peft.__version__)
    # Preserve the fine-grained-token workaround without printing the token.
    token_path = Path(HF_TOKEN_PATH)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        os.chmod(token_path, 0o600)
        stream.write(token)
    snapshot_download(BASE_MODEL, local_files_only=False)
    print("Base model reachable. This checks read access, not upload permission.")
    (workspace / "logs").mkdir(parents=True, exist_ok=True)
    print("triton", triton.__version__)
    if not shutil.which("g++"):
        raise RuntimeError("g++ is missing; torch.compile needs a C++ compiler")
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=True)
    print("Provision complete.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace/bh"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(f"Would check CUDA, install {PACKAGES} into {sys.executable}, cache HF_TOKEN privately, "
              f"check {BASE_MODEL} read access and verify compiler/GPU tools. No changes made.")
        return 0
    try:
        provision(args.workspace)
        return 0
    except Exception as exc:
        # Provider failures may include request data; do not echo secret-bearing text.
        print(f"Provisioning failed ({type(exc).__name__}). Check the environment, access and tools; no instance stop was requested.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
