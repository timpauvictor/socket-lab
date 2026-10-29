#!/usr/bin/env python3
"""Language box training on Modal: one LoRA per Wikipedia language.

Runs training/train_box.py for one language per invocation. Smoke first
(500 chunks, 20 steps, rank 8, head sampling), then full (100k chunks,
2 epochs, rank 16, reservoir sampling). Continuation probe per box;
manifest + verified local download follow the smoke pattern.

Launched with (repo root):
  modal run cloud/modal/train_box.py::app.smoke --lang ro
  modal run cloud/modal/train_box.py::app.full --lang ro
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
from cloud.modal.processes import package_versions, run_streaming

TRAIN_SCRIPT = REPO_ROOT / "training" / "train_box.py"

APP_NAME = "bespoke-hypernetworks-train-box"
VOLUME_NAME = "bespoke-hypernetworks-vault"
REMOTE_ROOT = "/adapters/qwen06b"

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
LANGS = ("ro", "en", "fr", "de")

PROBE_PREFIX = {
    "ro": "Bucuresti este capitala",
    "en": "London is the capital",
    "fr": "Paris est la capitale",
    "de": "Berlin ist die Hauptstadt",
}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "transformers>=4.57,<5", "peft", "datasets", "accelerate", "huggingface_hub")
    .add_local_file(str(Path(__file__).resolve()), "/root/job_definition.py")
    .add_local_file(str(TRAIN_SCRIPT), "/root/job_train_box.py")
    .add_local_file(str(REPO_ROOT / "cloud/modal/artifacts.py"), "/root/cloud/modal/artifacts.py")
    .add_local_file(str(REPO_ROOT / "cloud/modal/processes.py"), "/root/cloud/modal/processes.py")
)


@app.function(
    image=train_image,
    gpu="L40S",
    cpu=4,
    memory=16384,
    timeout=21600,
    volumes={"/vault": volume},
)
def train_run(run_id: str, lang: str, max_chunks: int, max_steps: int,
              rank: int, epochs: float, sample: str, scan_mult: float = 5.0) -> dict:
    out_dir = Path("/vault") / REMOTE_ROOT.lstrip("/") / f"box-{lang}" / run_id
    marker = out_dir / "_COMPLETE.json"

    command = [
        sys.executable, "/root/job_train_box.py",
        "--model", BASE_MODEL,
        "--output", str(out_dir),
        "--lang", lang,
        "--max_chunks", str(max_chunks),
        "--max_steps", str(max_steps),
        "--epochs", str(epochs),
        "--rank", str(rank),
        "--sample", sample,
        "--scan_mult", str(scan_mult),
    ]
    request = request_identity(command, sources=["/root/job_definition.py"])
    existing = check_run_request(out_dir, run_id, request)
    if existing is not None:
        return existing
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"Incomplete output exists, inspect before retry: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(out_dir / "_REQUEST.json", request)
    print(f"Starting box training: {' '.join(command)}", flush=True)
    stdout, stderr, returncode = run_streaming(command)
    (out_dir / "train.log").write_text(stdout + "\n" + stderr, encoding="utf-8")
    if returncode != 0:
        raise RuntimeError(f"train_box.py failed with exit {returncode}; see train.log")
    match = re.findall(r"'(?:train_)?loss':\s*([0-9]+\.[0-9]+)", stdout)
    final_loss = float(match[-1]) if match else None
    print(f"Final training loss: {final_loss}", flush=True)

    print(f"Running continuation probe for {lang} on base + box", flush=True)
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
        for prefix in (PROBE_PREFIX[lang],):
            enc = tok(prefix, return_tensors="pt").to("cuda")
            out = model.generate(**enc, max_new_tokens=30, do_sample=False)
            cont = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0].strip()
            probes.append({"prefix": prefix, "continuation": cont})
            print(f"PROBE prefix={prefix!r} continuation={cont!r}", flush=True)
    (out_dir / "probe.json").write_text(json.dumps(probes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    files = file_manifest(out_dir, ("adapter_config.json", "adapter_model.safetensors", "train.log", "probe.json"))
    result = {
        "status": "complete",
        "request": request,
        "run_id": run_id,
        "volume": VOLUME_NAME,
        "volume_path": f"{REMOTE_ROOT}/box-{lang}/{run_id}",
        "base_model": BASE_MODEL,
        "lang": lang,
        "max_chunks": max_chunks,
        "max_steps": max_steps,
        "epochs": epochs,
        "rank": rank,
        "sample": sample,
        "scan_mult": scan_mult,
        "final_loss": final_loss,
        "files": files,
        "versions": package_versions(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_manifest(marker, result)
    volume.commit()
    print(f"Committed manifest: {VOLUME_NAME}:{REMOTE_ROOT}/box-{lang}/{run_id}/_COMPLETE.json", flush=True)
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
    print("Box run complete.", flush=True)


@app.local_entrypoint()
def smoke(lang: str) -> None:
    if lang not in LANGS:
        raise RuntimeError(f"Unknown language: {lang}; use one of {LANGS}")
    run_id = f"qwen06b-box-{lang}-smoke-001"
    _retrieve(train_run.remote(run_id, lang, 500, 20, 8, 2.0, "head"), f"box-{lang}-smoke")


@app.local_entrypoint()
def full(lang: str) -> None:
    if lang not in LANGS:
        raise RuntimeError(f"Unknown language: {lang}; use one of {LANGS}")
    run_id = f"qwen06b-box-{lang}-full-001"
    _retrieve(train_run.remote(run_id, lang, 100000, 0, 16, 2.0, "reservoir"), f"box-{lang}")


def _run_all(max_chunks: int, max_steps: int, rank: int, epochs: float,
             sample: str, tag: str) -> None:
    """One app run, one train_run call per language (fan-out)."""
    specs = [
        ((f"qwen06b-box-{lang}-{tag}-001", lang, max_chunks, max_steps,
          rank, epochs, sample),
         f"box-{lang}" + ("-smoke" if tag == "smoke" else ""))
        for lang in LANGS
    ]
    results = list(train_run.starmap([args for args, _ in specs],
                                     return_exceptions=True))
    failures = {}
    for (args, subdir), result in zip(specs, results):
        if isinstance(result, Exception):
            failures[args[1]] = result
        else:
            _retrieve(result, subdir)
    if failures:
        detail = "; ".join(f"{lang}: {err!r}" for lang, err in failures.items())
        raise RuntimeError(f"Box fan-out had {len(failures)} failure(s): {detail}")
    print(f"All {len(specs)} box runs complete.", flush=True)


@app.local_entrypoint()
def smoke_all() -> None:
    _run_all(500, 20, 8, 2.0, "head", "smoke")


@app.local_entrypoint()
def full_all() -> None:
    _run_all(100000, 0, 16, 2.0, "reservoir", "full")
