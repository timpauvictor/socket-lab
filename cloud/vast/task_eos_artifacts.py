"""Frozen inputs and hash-verified nested bundle for one controlled EOS run."""
import argparse
import json
import platform
import random
import re
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloud.modal.artifacts import file_manifest, hash_file, write_manifest
from evaluation.eval_llama import load_pairs
from training.task_model import validate_adapter
from training.translation_data import find_config, prepare_pairs

MODEL = "Qwen/Qwen3-0.6B-Base"
PAIRS = ("ro-en", "fr-en", "de-en")
DIRECTIONS = ("en-ro", "ro-en", "de-en")
ADAPTERS = ("box-ro", "box-en", "box-fr", "box-de", "task-full", "eos-tune-001")
SOURCES = ("training/train_task.py", "training/translation_data.py", "training/task_model.py",
           "evaluation/eval_task_eos.py", "evaluation/eval_llama.py", "evaluation/prompts.py",
           "evaluation/repetition.py", "cloud/vast/task_eos_experiment.py",
           "cloud/vast/task_eos_artifacts.py", "cloud/vast/run.py", "cloud/vast/recipes.py",
           "cloud/vast/lifecycle.py", "cloud/modal/artifacts.py", "cloud/vast/task-eos-requirements.txt")
SETTINGS = {"model": MODEL, "pairs": PAIRS, "steps": [333, 333, 334], "seed": 42,
            "learning_rate": 5e-5, "microbatch": 8, "accumulation": 4, "rank": 32,
            "bf16": True, "max_length": 256, "content_budget": 255, "warmup_ratio": 0.05,
            "scheduler": "cosine_per_pair_restart", "compile": False,
            "max_pairs": 150000, "composition": "peft-linear-1+1", "max_new_tokens": 128}


def input_names() -> list[str]:
    return [f"adapters/{a}/{n}" for a in ADAPTERS for n in ("adapter_config.json", "adapter_model.safetensors")] + [
        f"logs/eval-{d}-devtest.jsonl" for d in DIRECTIONS]


def check_revision(revision: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Use a resolved 40-hex Hub commit, not main or a branch/tag")
    return revision


def runtime_metadata() -> dict:
    packages = {}
    for name in ("torch", "transformers", "peft", "datasets", "accelerate", "sacrebleu", "huggingface_hub"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    metadata = {"python": sys.version, "platform": platform.platform(), "packages": packages}
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required")
    metadata["hardware"] = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                            "vram_bytes": torch.cuda.get_device_properties(0).total_memory}
    if packages["sacrebleu"] != "2.6.0":
        raise RuntimeError("sacrebleu==2.6.0 is required")
    return metadata


def validate_provenance_entry(repo: Path, adapter: str, entry: dict) -> None:
    if not entry.get("source") or not entry.get("evidence"):
        raise ValueError(f"Source inspection evidence required for {adapter}")
    if entry.get("source_type", "hub") == "modal_manifest":
        if not adapter.startswith("box-"):
            raise ValueError("Only original language boxes use Modal completion provenance")
        path = repo / "adapters" / adapter / "_COMPLETE.json"
        if hash_file(path)[0] != entry.get("manifest_sha256"):
            raise ValueError(f"Completion manifest hash mismatch: {adapter}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected_source = f"modal-volume:{manifest['volume']}:{manifest['volume_path']}"
        if (manifest.get("status") != "complete" or manifest.get("base_model") != MODEL
                or manifest.get("rank") != 16 or manifest.get("lang") != adapter.removeprefix("box-")
                or manifest.get("run_id") != entry.get("run_id") or entry["source"] != expected_source):
            raise ValueError(f"Incompatible completed-box provenance: {adapter}")
        recorded = {item["name"]: item["sha256"] for item in manifest["files"]}
        if any(recorded.get(name) != digest for name, digest in entry["sha256"].items()):
            raise ValueError(f"Completion manifest file identity mismatch: {adapter}")
    elif entry.get("source_type", "hub") == "hub":
        check_revision(entry["revision"])
        if adapter == "eos-tune-001" and entry["revision"] != "47a09863e61d03e0ef5bbae12cd36516ac770446":
            if not entry.get("approved_alternate_source"):
                raise ValueError("Monolith source differs without approved alternate-source evidence")
    else:
        raise ValueError("Unknown provenance source type")
    validate_adapter(repo / "adapters" / adapter, MODEL, 32 if adapter in ("task-full", "eos-tune-001") else 16)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if hash_file(repo / "adapters" / adapter / name)[0] != entry["sha256"][name]:
            raise ValueError(f"Provenance hash mismatch: {adapter}/{name}")


def freeze_inputs(repo: Path, output: Path, model_revision: str, dataset_revision: str,
                  provenance_path: Path) -> None:
    """Copy original inputs once, reject unverified/changed sources, select data once.

    Provenance is supplied after source inspection, not inferred from presence.
    Each adapter entry binds either a Hub commit or the original Modal completion
    manifest, inspection evidence and both file hashes. This is not an independent
    authenticity audit of the historical completion manifest.
    """
    check_revision(model_revision)
    check_revision(dataset_revision)
    if output.exists():
        raise ValueError("Frozen input destination must be fresh")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for adapter in ADAPTERS:
        validate_provenance_entry(repo, adapter, provenance[adapter])
    for direction in DIRECTIONS:
        src, tgt = direction.split("-")
        rows = load_pairs(SimpleNamespace(pairs=repo / f"logs/eval-{direction}-devtest.jsonl",
                                         src=src, tgt=tgt, split="devtest"))
        if len(rows) != 1012:
            raise ValueError(f"Expected 1,012 pairs for {direction}")
    metadata = runtime_metadata()
    output.mkdir(parents=True, exist_ok=False)
    for name in input_names():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, destination)
        if name.startswith("adapters/"):
            _, adapter, filename = name.split("/")
            if hash_file(destination)[0] != provenance[adapter]["sha256"][filename]:
                raise RuntimeError(f"Input changed while freezing: {name}")
    for name in ("train.log", "run-info.json", "manifest.json", "_COMPLETE.json"):
        for adapter in ADAPTERS:
            source = repo / "adapters" / adapter / name
            if source.is_file():
                destination = output / "adapters" / adapter / name
                shutil.copyfile(source, destination)
                if name == "_COMPLETE.json" and provenance[adapter].get("source_type") == "modal_manifest":
                    if hash_file(destination)[0] != provenance[adapter]["manifest_sha256"]:
                        raise RuntimeError(f"Completion provenance changed while freezing: {adapter}")
    (output / "data").mkdir()
    from datasets import get_dataset_config_names, load_dataset
    rng = random.Random(42)
    configs = get_dataset_config_names("Helsinki-NLP/opus-100", revision=dataset_revision)
    selections = {}
    for pair in PAIRS:
        src, tgt = pair.split("-")
        config = find_config(src, tgt, configs)
        dataset = load_dataset("Helsinki-NLP/opus-100", config, split="train", revision=dataset_revision)
        rows = prepare_pairs(dataset, src, tgt, 150000, rng, progress_every=100000)
        if not rows:
            raise ValueError(f"No usable training data: {pair}")
        write_manifest(output / "data" / f"{pair}.json", rows)
        selections[pair] = {"config": config, "selected_pairs": len(rows),
                            "dataset_fingerprint": getattr(dataset, "_fingerprint", None)}
    write_manifest(output / "identity.json", {"settings": SETTINGS, "model_revision": model_revision,
        "tokenizer_revision": model_revision, "dataset_revision": dataset_revision,
        "dataset": "Helsinki-NLP/opus-100", "selections": selections, "provenance": provenance,
        "runtime": metadata, "source_files": file_manifest(repo, SOURCES)})
    names = sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file())
    write_manifest(output / "manifest.json", {"files": file_manifest(output, names)})
    verify_bundle(output)


def verify_bundle(root: Path, *, exact: bool = True) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    names = []
    for entry in manifest["files"]:
        name = entry["name"]
        path = root / name
        if (not isinstance(name, str) or "\\" in name or name.startswith("/")
                or path.resolve().is_relative_to(root.resolve()) is False or path.is_symlink()):
            raise ValueError("Unsafe manifest path")
        if name in names or name == "manifest.json":
            raise ValueError("Duplicate/self manifest entry")
        names.append(name)
        if hash_file(path) != (entry["sha256"], entry["bytes"]):
            raise RuntimeError(f"Bundle hash mismatch: {name}")
    if exact:
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        if actual != set(names) | {"manifest.json"}:
            raise RuntimeError("Unlisted/missing bundle files")
    return manifest


def check_frozen(inputs: Path, repo: Path) -> None:
    verify_bundle(inputs)
    identity = json.loads((inputs / "identity.json").read_text(encoding="utf-8"))
    if identity["source_files"] != file_manifest(repo, SOURCES):
        raise RuntimeError("Experiment code drift since input freeze")


def seal_bundle(root: Path) -> dict:
    names = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    if "manifest.json" in names:
        raise ValueError("Bundle already sealed; inspect before retry")
    manifest = {"format": "controlled-task-eos-v1", "files": file_manifest(root, names)}
    write_manifest(root / "manifest.json", manifest)
    verify_bundle(root)
    return manifest


def upload_bundle(root: Path, destination: str) -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    verify_bundle(root)
    api = HfApi()
    # A new private repository is mandatory. No historical destination is reused.
    api.create_repo(destination, repo_type="model", private=True, exist_ok=False)
    commit = api.upload_folder(folder_path=str(root), repo_id=destination, repo_type="model")
    revision = commit.oid
    expected = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    remote = set(api.list_repo_files(destination, revision=revision))
    if not expected <= remote:
        raise RuntimeError("Uploaded bundle is missing files")
    for name in sorted(expected):
        downloaded = Path(hf_hub_download(destination, name, revision=revision))
        if hash_file(downloaded) != hash_file(root / name):
            raise RuntimeError(f"Remote bundle hash mismatch: {name}")
    return {"destination": destination, "revision": revision, "files_verified": len(expected),
            "manifest_sha256": hash_file(root / "manifest.json")[0]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--repo", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--model-revision", required=True)
    freeze.add_argument("--dataset-revision", required=True)
    freeze.add_argument("--provenance", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("root", type=Path)
    final = sub.add_parser("finalize")
    final.add_argument("root", type=Path)
    final.add_argument("destination")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze_inputs(args.repo, args.output, args.model_revision, args.dataset_revision, args.provenance)
    elif args.command == "verify":
        verify_bundle(args.root)
        print("All listed bundle hashes verified; no files changed.")
    else:
        seal_bundle(args.root)
        print(json.dumps(upload_bundle(args.root, args.destination)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
