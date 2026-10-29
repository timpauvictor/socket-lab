#!/usr/bin/env python3
"""Task cartridge training on Modal: translating skill over frozen boxes.

Runs training/train_task.py, which merge-cycles ro-en, fr-en, de-en: per
pair it merges that pair's frozen box adapters into the base, attaches the
shared task adapter, and trains the loader-derived K steps. Smoke uses the
rank-8 smoke boxes (shapes must match); full uses the rank-16 boxes.

Before launch, check that every required box has a manifest marked complete.
If any is missing or incomplete, the run refuses to start.

Launched with (repo root):
  modal run cloud/modal/train_task.py::app.smoke
  modal run cloud/modal/train_task.py::app.full
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

TRAIN_SCRIPT = REPO_ROOT / "training" / "train_task.py"

APP_NAME = "bespoke-hypernetworks-train-task"
VOLUME_NAME = "bespoke-hypernetworks-vault"
REMOTE_ROOT = "/adapters/qwen06b"

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
PAIRS = "ro-en,fr-en,de-en"
LANGS = ("ro", "en", "fr", "de")

SMOKE_RUN_ID = "qwen06b-task-smoke-001"
FULL_RUN_ID = "qwen06b-task-full-001"

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
    .add_local_file(str(TRAIN_SCRIPT), "/root/job_train_task.py")
    .add_local_file(str(REPO_ROOT / "training/translation_data.py"), "/root/translation_data.py")
    .add_local_file(str(REPO_ROOT / "training/task_model.py"), "/root/task_model.py")
    .add_local_file(str(REPO_ROOT / "cloud/modal/artifacts.py"), "/root/cloud/modal/artifacts.py")
    .add_local_file(str(REPO_ROOT / "cloud/modal/processes.py"), "/root/cloud/modal/processes.py")
)


def _box_run_id(lang: str, phase: str) -> str:
    return f"qwen06b-box-{lang}-{phase}-001"


def _box_dir(lang: str, phase: str) -> str:
    return f"{REMOTE_ROOT}/box-{lang}/{_box_run_id(lang, phase)}"


def _require_box_manifests(phase: str) -> None:
    for lang in LANGS:
        marker = f"{_box_dir(lang, phase)}/_COMPLETE.json"
        try:
            raw = b"".join(volume.read_file(marker))
        except FileNotFoundError:
            raise RuntimeError(f"Required box manifest missing, train boxes first: {marker}")
        manifest = json.loads(raw)
        if manifest.get("status") != "complete":
            raise RuntimeError(f"Box manifest not complete, train boxes first: {marker}")
    print(f"All 4 box manifests verified ({phase})", flush=True)


@app.function(
    image=train_image,
    gpu="L40S",
    cpu=4,
    memory=16384,
    timeout=21600,
    volumes={"/vault": volume},
)
def train_run(run_id: str, phase: str, max_pairs: int, max_steps: int,
              rank: int, epochs: float) -> dict:
    out_dir = Path("/vault") / REMOTE_ROOT.lstrip("/") / "task" / run_id
    marker = out_dir / "_COMPLETE.json"

    command = [
        sys.executable, "/root/job_train_task.py",
        "--model", BASE_MODEL,
        "--output", str(out_dir),
    ]
    for language in LANGS:
        command += ["--box", f"{language}=/vault{_box_dir(language, phase)}"]
    command += [
        "--pairs", PAIRS,
        "--max_pairs", str(max_pairs),
        "--max_steps", str(max_steps),
        "--epochs", str(epochs),
        "--rank", str(rank),
    ]
    inputs = [f"/vault{_box_dir(lang, phase)}/{name}" for lang in LANGS
              for name in ("adapter_config.json", "adapter_model.safetensors")]
    request = request_identity(command, sources=["/root/translation_data.py", "/root/task_model.py", "/root/job_definition.py"], inputs=inputs)
    existing = check_run_request(out_dir, run_id, request)
    if existing is not None:
        return existing
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"Incomplete output exists, inspect before retry: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(out_dir / "_REQUEST.json", request)
    print(f"Starting task training: {' '.join(command)}", flush=True)
    stdout, stderr, returncode = run_streaming(command)
    (out_dir / "train.log").write_text(stdout + "\n" + stderr, encoding="utf-8")
    if returncode != 0:
        raise RuntimeError(f"train_task.py failed with exit {returncode}; see train.log")
    match = re.findall(r"Pair \S+: \d+ examples", stdout)
    print(f"Trained pairs: {len(match)}", flush=True)

    print("Running probe on merged boxes + task adapter", flush=True)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Probe under the ro-en background, matching the first merge cycle.
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    base = PeftModel.from_pretrained(base, f"/vault{_box_dir('ro', phase)}", adapter_name="bg0")
    base.load_adapter(f"/vault{_box_dir('en', phase)}", adapter_name="bg1")
    base.add_weighted_adapter(["bg0", "bg1"], [1.0, 1.0], adapter_name="combo", combination_type="linear")
    base.set_adapter("combo")
    merged = base.merge_and_unload().to("cuda")
    model = PeftModel.from_pretrained(merged, str(out_dir))
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
        "volume": VOLUME_NAME,
        "volume_path": f"{REMOTE_ROOT}/task/{run_id}",
        "base_model": BASE_MODEL,
        "pairs": PAIRS,
        "box_phase": phase,
        "max_pairs": max_pairs,
        "max_steps": max_steps,
        "epochs": epochs,
        "rank": rank,
        "files": files,
        "versions": package_versions(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_manifest(marker, result)
    volume.commit()
    print(f"Committed manifest: {VOLUME_NAME}:{REMOTE_ROOT}/task/{run_id}/_COMPLETE.json", flush=True)
    return result


def _retrieve(manifest: dict, local_subdir: str) -> None:
    if manifest.get("status") != "complete":
        raise RuntimeError("Remote run did not complete")
    local_dir = REPO_ROOT / "adapters" / local_subdir
    local_dir.mkdir(parents=True, exist_ok=True)
    retrieve_files(manifest, local_dir, volume.read_file,
                   ("adapter_config.json", "adapter_model.safetensors"))
    write_manifest(local_dir / "_COMPLETE.json", manifest)
    print("Task run complete.", flush=True)


@app.local_entrypoint()
def smoke() -> None:
    _require_box_manifests("smoke")
    _retrieve(train_run.remote(SMOKE_RUN_ID, "smoke", 500, 20, 8, 2.0), "task-smoke")


@app.local_entrypoint()
def full() -> None:
    _require_box_manifests("full")
    _retrieve(train_run.remote(FULL_RUN_ID, "full", 150000, 0, 32, 2.0), "task")
