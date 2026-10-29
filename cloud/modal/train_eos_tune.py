#!/usr/bin/env python3
"""EOS termination tune-up on the finished monolith adapter, serverless.

Reuses the finished local monolith adapter (baked into the image) as
--tune_from source, trains with --eos_stop targets, then runs the
greedy termination probe (base vs tuned) on-GPU. Short job:
smoke = 2 steps, full = 1000 steps at LR 5e-5. No --compile on purpose.

Launched with (repo root):
  modal run cloud/modal/train_eos_tune.py::app.smoke
  modal run cloud/modal/train_eos_tune.py::app.full
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from cloud.modal.artifacts import check_run_request, request_identity, file_manifest, retrieve_files, write_manifest
from cloud.modal.processes import package_versions, run_streaming

TRAIN_SCRIPT = REPO_ROOT / "training" / "train_lora.py"
PROBE_SCRIPT = REPO_ROOT / "evaluation" / "probe_terminate.py"
TUNE_SRC_CONFIG = REPO_ROOT / "adapters" / "monolith" / "adapter_config.json"
TUNE_SRC_WEIGHTS = REPO_ROOT / "adapters" / "monolith" / "adapter_model.safetensors"

APP_NAME = "bespoke-qwen06b-train-eos-tune"
VOLUME_NAME = "bespoke-qwen06b-vault"
REMOTE_ROOT = "/adapters/qwen06b/eos-tune"

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
PAIRS = "ro-en,fr-en,de-en"

SMOKE_RUN_ID = "qwen06b-eos-smoke-001"
FULL_RUN_ID = "qwen06b-eos-tune-001"

TUNE_SRC = "/root/tune-src"
LR = 5e-5

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "transformers>=4.57,<5", "peft", "datasets", "accelerate", "huggingface_hub")
    .add_local_file(str(Path(__file__).resolve()), "/root/job_definition.py")
    .add_local_file(str(TRAIN_SCRIPT), "/root/train_lora.py")
    .add_local_file(str(REPO_ROOT / "training/translation_data.py"), "/root/translation_data.py")
    .add_local_file(str(PROBE_SCRIPT), "/root/probe_terminate.py")
    .add_local_file(str(REPO_ROOT / "evaluation/prompts.py"), "/root/prompts.py")
    .add_local_file(str(REPO_ROOT / "evaluation/repetition.py"), "/root/repetition.py")
    .add_local_file(str(TUNE_SRC_CONFIG), f"{TUNE_SRC}/adapter_config.json")
    .add_local_file(str(TUNE_SRC_WEIGHTS), f"{TUNE_SRC}/adapter_model.safetensors")
    .add_local_file(str(REPO_ROOT / "cloud/modal/artifacts.py"), "/root/cloud/modal/artifacts.py")
    .add_local_file(str(REPO_ROOT / "cloud/modal/processes.py"), "/root/cloud/modal/processes.py")
)


@app.function(
    image=train_image,
    gpu="L40S",
    cpu=4,
    memory=16384,
    timeout=7200,
    volumes={"/vault": volume},
)
def tune_run(run_id: str, max_steps: int, lr: float) -> dict:
    out_dir = Path("/vault") / REMOTE_ROOT.lstrip("/") / run_id
    marker = out_dir / "_COMPLETE.json"


    command = [
        sys.executable, "/root/train_lora.py",
        "--model", BASE_MODEL,
        "--output", str(out_dir),
        "--pairs", PAIRS,
        "--max_pairs", "150000",
        "--epochs", "2.0",
        "--rank", "32",
        "--lr", str(lr),
        "--max_steps", str(max_steps),
        "--eos_stop",
        "--tune_from", TUNE_SRC,
    ]
    inputs = [f"{TUNE_SRC}/{name}" for name in ("adapter_config.json", "adapter_model.safetensors")]
    request = request_identity(command, sources=["/root/translation_data.py", "/root/job_definition.py",
                                                 "/root/probe_terminate.py", "/root/prompts.py",
                                                 "/root/repetition.py"], inputs=inputs)
    existing = check_run_request(out_dir, run_id, request)
    if existing is not None:
        return existing
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"Incomplete output exists, inspect before retry: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(out_dir / "_REQUEST.json", request)
    print(f"Starting EOS tune-up: {' '.join(command)}", flush=True)
    stdout, stderr, returncode = run_streaming(command)
    (out_dir / "train.log").write_text(stdout + "\n" + stderr, encoding="utf-8")
    if returncode != 0:
        raise RuntimeError(f"train_lora.py failed with exit {returncode}; see train.log")
    match = re.findall(r"'(?:train_)?loss':\s*([0-9]+\.[0-9]+)", stdout)
    final_loss = float(match[-1]) if match else None
    print(f"Final training loss: {final_loss}", flush=True)

    print("Running termination probe: base, then tuned adapter", flush=True)
    base_out, _, base_rc = run_streaming([sys.executable, "/root/probe_terminate.py"])
    tuned_out, _, tuned_rc = run_streaming(
        [sys.executable, "/root/probe_terminate.py", "--adapter", str(out_dir)])
    if base_rc != 0 or tuned_rc != 0:
        raise RuntimeError(f"probe failed (base rc={base_rc}, tuned rc={tuned_rc})")
    (out_dir / "probe-base.txt").write_text(base_out, encoding="utf-8")
    (out_dir / "probe-tuned.txt").write_text(tuned_out, encoding="utf-8")

    files = file_manifest(out_dir, ("adapter_config.json", "adapter_model.safetensors",
                                    "train.log", "probe-base.txt", "probe-tuned.txt"))
    result = {
        "status": "complete",
        "request": request,
        "run_id": run_id,
        "tune_from": "local adapters/monolith (baked into image)",
        "volume": VOLUME_NAME,
        "volume_path": f"{REMOTE_ROOT}/{run_id}",
        "base_model": BASE_MODEL,
        "pairs": PAIRS,
        "max_steps": max_steps,
        "lr": lr,
        "eos_stop": True,
        "rank": 32,
        "final_loss": final_loss,
        "files": files,
        "versions": package_versions(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_manifest(marker, result)
    volume.commit()
    print(f"Committed manifest: {VOLUME_NAME}:{REMOTE_ROOT}/{run_id}/_COMPLETE.json", flush=True)
    return result


def _retrieve(manifest: dict, local_subdir: str) -> None:
    if manifest.get("status") != "complete":
        raise RuntimeError("Remote run did not complete")
    print(f"final_loss={manifest.get('final_loss')}", flush=True)
    local_dir = REPO_ROOT / "adapters" / local_subdir
    local_dir.mkdir(parents=True, exist_ok=True)
    retrieve_files(manifest, local_dir, volume.read_file,
                   ("adapter_config.json", "adapter_model.safetensors",
                    "train.log", "probe-base.txt", "probe-tuned.txt"))
    write_manifest(local_dir / "_COMPLETE.json", manifest)
    print("EOS tune-up run complete.", flush=True)


@app.local_entrypoint()
def smoke() -> None:
    _retrieve(tune_run.remote(SMOKE_RUN_ID, 2, LR), "eos-tune-smoke")


@app.local_entrypoint()
def full() -> None:
    _retrieve(tune_run.remote(FULL_RUN_ID, 1000, LR), "eos-tune-001")
