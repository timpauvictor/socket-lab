#!/usr/bin/env python3
"""Smoke-test the opus-100 LoRA training loop on Modal (L40S, ~2 minutes).

Runs training/train_lora.py UNMODIFIED with the smoke config
(500 pairs, 20 steps, rank 8), saves a 3-sample probe plus a _COMPLETE.json
manifest to the Modal volume, and downloads the verified adapter locally.

Launched with: modal run cloud/modal/train_smoke.py::app.smoke
Working directory must be the repo root (train script is mounted by path).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from cloud.modal.artifacts import check_run_request, request_identity, file_manifest, retrieve_files, write_manifest
from cloud.modal.processes import pip_freeze
TRAIN_SCRIPT = REPO_ROOT / "training" / "train_lora.py"

APP_NAME = "bespoke-hypernetworks-train-smoke"
VOLUME_NAME = "bespoke-hypernetworks-vault"
RUN_ID = "qwen06b-roen-smoke-003"  # bumped: fixed missing padding in train_lora.py
REMOTE_DIR = f"/adapters/qwen06b-roen/{RUN_ID}"
COMPLETE_FILE = f"{REMOTE_DIR}/_COMPLETE.json"

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
PAIRS = "ro-en"
SMOKE_MAX_PAIRS = 500
SMOKE_MAX_STEPS = 20
SMOKE_RANK = 8

PROBE_SENTENCES = [
    "Buna dimineata.",
    "Multumesc pentru ajutor.",
    "Vremea este frumoasa astazi.",
]

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "transformers>=4.57,<5", "peft", "datasets", "accelerate", "huggingface_hub")
    .add_local_file(str(Path(__file__).resolve()), "/root/job_definition.py")
    .add_local_file(str(TRAIN_SCRIPT), "/root/train_lora.py")
    .add_local_file(str(REPO_ROOT / "training/translation_data.py"), "/root/translation_data.py")
    .add_local_file(str(REPO_ROOT / "cloud" / "modal" / "artifacts.py"), "/root/cloud/modal/artifacts.py")
    .add_local_file(str(REPO_ROOT / "cloud" / "modal" / "processes.py"), "/root/cloud/modal/processes.py")
)


@app.function(
    image=train_image,
    gpu="L40S",
    cpu=4,
    memory=16384,
    timeout=1800,
    volumes={"/vault": volume},
)
def train_smoke() -> dict:
    out_dir = Path("/vault") / REMOTE_DIR.lstrip("/")
    marker = Path("/vault") / COMPLETE_FILE.lstrip("/")

    command = [
        sys.executable, "/root/train_lora.py",
        "--model", BASE_MODEL,
        "--output", str(out_dir),
        "--pairs", PAIRS,
        "--max_pairs", str(SMOKE_MAX_PAIRS),
        "--max_steps", str(SMOKE_MAX_STEPS),
        "--rank", str(SMOKE_RANK),
    ]
    request = request_identity(command, sources=["/root/translation_data.py", "/root/job_definition.py"])
    existing = check_run_request(out_dir, RUN_ID, request)
    if existing is not None:
        return existing
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"Incomplete output exists, inspect before retry: {REMOTE_DIR}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(out_dir / "_REQUEST.json", request)
    print(f"Starting smoke training: {' '.join(command)}", flush=True)
    proc = subprocess.run(command, capture_output=True, text=True)
    (out_dir / "train.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    print(proc.stdout, flush=True)
    if proc.returncode != 0:
        print(proc.stderr, flush=True)
        raise RuntimeError(f"train_lora.py failed with exit {proc.returncode}; see train.log")
    match = re.findall(r"'(?:train_)?loss':\s*([0-9]+\.[0-9]+)", proc.stdout)
    final_loss = float(match[-1]) if match else None
    print(f"Final training loss: {final_loss}", flush=True)

    print("Running 3-sample probe on base + smoke adapter", flush=True)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, str(out_dir))
    model.eval()
    probes = []
    with torch.inference_mode():
        for sentence in PROBE_SENTENCES:
            prompt = f"Translate the following Romanian text to English.\nRomanian: {sentence}\nEnglish: "
            enc = tok(prompt, return_tensors="pt").to("cuda")
            out = model.generate(**enc, max_new_tokens=64, do_sample=False)
            hyp = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0].strip()
            probes.append({"src": sentence, "hyp": hyp})
            print(f"PROBE src={sentence!r} hyp={hyp!r}", flush=True)
    (out_dir / "probe.json").write_text(json.dumps(probes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    files = file_manifest(out_dir, ("adapter_config.json", "adapter_model.safetensors", "train.log", "probe.json"))
    result = {
        "status": "complete",
        "request": request,
        "phase": "smoke",
        "run_id": RUN_ID,
        "volume": VOLUME_NAME,
        "volume_path": REMOTE_DIR,
        "base_model": BASE_MODEL,
        "pairs": PAIRS,
        "max_pairs": SMOKE_MAX_PAIRS,
        "max_steps": SMOKE_MAX_STEPS,
        "rank": SMOKE_RANK,
        "final_loss": final_loss,
        "files": files,
        "versions": pip_freeze(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_manifest(marker, result)
    volume.commit()
    print(f"Committed smoke manifest: {VOLUME_NAME}:{COMPLETE_FILE}", flush=True)
    return result


@app.local_entrypoint()
def smoke() -> None:
    manifest = train_smoke.remote()
    if manifest.get("status") != "complete":
        raise RuntimeError("Remote smoke run did not complete")
    print(f"final_loss={manifest.get('final_loss')}", flush=True)

    local_dir = REPO_ROOT / "adapters" / "smoke"
    local_dir.mkdir(parents=True, exist_ok=True)
    retrieve_files(manifest, local_dir, volume.read_file, ("adapter_config.json", "adapter_model.safetensors"), remote_path=REMOTE_DIR)
    write_manifest(local_dir / "_COMPLETE.json", manifest)
    print("Smoke complete. Adapter ready for eval_translate.py --adapters.", flush=True)
