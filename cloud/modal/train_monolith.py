#!/usr/bin/env python3
"""Monolith control: one LoRA on all opus-100 pairs mixed (ro/fr/de-en).

Runs training/train_lora.py with --pairs ro-en,fr-en,de-en. Smoke first
(500 pairs each, 20 steps, rank 8), then full (150k pairs each, 2 epochs,
rank 32). Probe covers one direction per pair; manifest + verified local
download follow the smoke pattern.

Launched with (repo root):
  modal run cloud/modal/train_monolith.py::app.smoke
  modal run cloud/modal/train_monolith.py::app.full
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from cloud.modal.artifacts import check_run_request, request_identity, file_manifest, retrieve_files, write_manifest
from cloud.modal.processes import pip_freeze, run_streaming

TRAIN_SCRIPT = REPO_ROOT / "training" / "train_lora.py"

APP_NAME = "bespoke-hypernetworks-train-monolith"
VOLUME_NAME = "bespoke-hypernetworks-vault"
REMOTE_ROOT = "/adapters/qwen06b/monolith"

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
PAIRS = "ro-en,fr-en,de-en"

SMOKE_RUN_ID = "qwen06b-monolith-smoke-001"
FULL_RUN_ID = "qwen06b-monolith-full-001"

PROBES = [
    ("ro", "Buna dimineata."),
    ("fr", "Bonjour, comment allez-vous ?"),
    ("de", "Guten Morgen, wie geht es dir?"),
]
LANG_NAMES = {"ro": "Romanian", "fr": "French", "de": "German"}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "transformers>=4.57,<5", "peft", "datasets", "accelerate", "huggingface_hub")
    .add_local_file(str(Path(__file__).resolve()), "/root/job_definition.py")
    .add_local_file(str(TRAIN_SCRIPT), "/root/train_lora.py")
    .add_local_file(str(REPO_ROOT / "training/translation_data.py"), "/root/translation_data.py")
    .add_local_file(str(REPO_ROOT / "cloud/modal/artifacts.py"), "/root/cloud/modal/artifacts.py")
    .add_local_file(str(REPO_ROOT / "cloud/modal/processes.py"), "/root/cloud/modal/processes.py")
)


# Weight files Trainer 4.x accepts as a valid checkpoint (trainer.py
# _load_from_checkpoint). An interrupted save can leave a checkpoint-*
# dir with no weights; such dirs must be skipped, not resumed from.
_WEIGHT_FILES = ("pytorch_model.bin", "model.safetensors",
                 "pytorch_model.bin.index.json", "model.safetensors.index.json",
                 "adapter_model.bin", "adapter_model.safetensors")


def _valid_checkpoints(out_dir: Path) -> list[Path]:
    """Checkpoint dirs newest-last, keeping only ones Trainer can load."""
    found = []
    for path in out_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        if any((path / name).is_file() for name in _WEIGHT_FILES):
            found.append((step, path))
    return [path for _, path in sorted(found)]


@app.function(
    image=train_image,
    gpu="L40S",
    cpu=4,
    memory=16384,
    timeout=43200,
    volumes={"/vault": volume},
)
def train_run(run_id: str, max_pairs: int, max_steps: int, rank: int, epochs: float) -> dict:
    out_dir = Path("/vault") / REMOTE_ROOT.lstrip("/") / run_id
    marker = out_dir / "_COMPLETE.json"

    command = [
        sys.executable, "/root/train_lora.py",
        "--model", BASE_MODEL,
        "--output", str(out_dir),
        "--pairs", PAIRS,
        "--max_pairs", str(max_pairs),
        "--max_steps", str(max_steps),
        "--epochs", str(epochs),
        "--rank", str(rank),
    ]
    request = request_identity(command, sources=["/root/translation_data.py", "/root/job_definition.py"])
    existing = check_run_request(out_dir, run_id, request)
    if existing is not None:
        return existing
    resume_from = ""
    if out_dir.exists() and any(out_dir.iterdir()):
        valid = _valid_checkpoints(out_dir)
        if not valid:
            raise RuntimeError(f"Incomplete output exists, inspect before retry: {out_dir}")
        resume_from = str(valid[-1])
        print(f"Resuming monolith run from {resume_from}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(out_dir / "_REQUEST.json", request)
    if resume_from:
        command += ["--resume_from", resume_from]
    print(f"Starting monolith training: {' '.join(command)}", flush=True)
    stdout, stderr, returncode = run_streaming(command)
    (out_dir / "train.log").write_text(stdout + "\n" + stderr, encoding="utf-8")
    if returncode != 0:
        raise RuntimeError(f"train_lora.py failed with exit {returncode}; see train.log")
    match = re.findall(r"'(?:train_)?loss':\s*([0-9]+\.[0-9]+)", stdout)
    final_loss = float(match[-1]) if match else None
    print(f"Final training loss: {final_loss}", flush=True)

    print("Running probe (one direction per pair) on base + adapter", flush=True)
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
        for lang, sentence in PROBES:
            prompt = (f"Translate the following {LANG_NAMES[lang]} text to English.\n"
                      f"{LANG_NAMES[lang]}: {sentence}\nEnglish: ")
            enc = tok(prompt, return_tensors="pt").to("cuda")
            out = model.generate(**enc, max_new_tokens=64, do_sample=False)
            hyp = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0].strip()
            probes.append({"lang": lang, "src": sentence, "hyp": hyp})
            print(f"PROBE {lang} src={sentence!r} hyp={hyp!r}", flush=True)
    (out_dir / "probe.json").write_text(json.dumps(probes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    files = file_manifest(out_dir, ("adapter_config.json", "adapter_model.safetensors", "train.log", "probe.json"))
    result = {
        "status": "complete",
        "request": request,
        "run_id": run_id,
        "resumed_from": resume_from,
        "volume": VOLUME_NAME,
        "volume_path": f"{REMOTE_ROOT}/{run_id}",
        "base_model": BASE_MODEL,
        "pairs": PAIRS,
        "max_pairs": max_pairs,
        "max_steps": max_steps,
        "epochs": epochs,
        "rank": rank,
        "final_loss": final_loss,
        "files": files,
        "versions": pip_freeze(),
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
                   ("adapter_config.json", "adapter_model.safetensors"))
    write_manifest(local_dir / "_COMPLETE.json", manifest)
    print("Monolith run complete.", flush=True)


@app.local_entrypoint()
def smoke() -> None:
    _retrieve(train_run.remote(SMOKE_RUN_ID, 500, 20, 8, 2.0), "monolith-smoke")


@app.local_entrypoint()
def full() -> None:
    _retrieve(train_run.remote(FULL_RUN_ID, 150000, 0, 32, 2.0), "monolith")
