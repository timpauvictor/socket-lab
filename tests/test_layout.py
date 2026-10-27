"""Check entry points and packaged dependencies without cloud/model access."""
import hashlib
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOBS = [
    ("box", "train_box.py", "/root/job_train_box.py"),
    ("task", "train_task.py", "/root/job_train_task.py"),
    ("monolith", "train_lora.py", "/root/train_lora.py"),
    ("smoke", "train_lora.py", "/root/train_lora.py"),
    ("eos_tune", "train_lora.py", "/root/train_lora.py"),
]


@pytest.mark.parametrize("script", [
    "evaluation/eval_llama.py", "evaluation/eval_offline.py",
    "evaluation/eval_pivot.py", "evaluation/eval_scratchpad.py",
    "evaluation/loop_check.py", "evaluation/sampling_probe.py",
    "reporting/build_board.py", "reporting/plot_board.py",
    "run_eval.py", "run_train.py", "serve.py", "cloud/vast/run.py",
    "cloud/vast/lifecycle.py", "cloud/vast/upload.py", "cloud/vast/provision.py",
    "evaluation/eval_task_eos.py", "cloud/vast/task_eos_experiment.py", "cloud/vast/task_eos_artifacts.py",
])
def test_script_help_from_another_directory(script, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / script), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.fixture
def modal_job(monkeypatch):
    def load(job):
        modal = Mock()
        image = modal.Image.debian_slim.return_value
        image.uv_pip_install.return_value = image
        image.add_local_file.return_value = image
        app = modal.App.return_value
        app.function.side_effect = lambda **kwargs: lambda function: function
        app.local_entrypoint.side_effect = lambda **kwargs: lambda function: function
        monkeypatch.setitem(sys.modules, "modal", modal)
        monkeypatch.setattr(sys, "path", sys.path.copy())
        namespace = runpy.run_path(str(ROOT / "cloud/modal" / f"train_{job}.py"))
        return namespace, image, modal.Volume.from_name.return_value
    return load


@pytest.mark.parametrize("job,trainer,remote", JOBS)
def test_modal_images_reference_relocated_sources(job, trainer, remote, modal_job, tmp_path):
    namespace, image, volume = modal_job(job)
    assert namespace["REPO_ROOT"] == ROOT
    image.add_local_file.assert_any_call(str(ROOT / "training" / trainer), remote)
    if job == "eos_tune":
        image.add_local_file.assert_any_call(str(ROOT / "evaluation/probe_terminate.py"), "/root/probe_terminate.py")
        assert namespace["TUNE_SRC_CONFIG"] == ROOT / "adapters/monolith/adapter_config.json"
        assert namespace["TUNE_SRC_WEIGHTS"] == ROOT / "adapters/monolith/adapter_model.safetensors"
    volume.commit.assert_not_called()
    volume.read_file.assert_not_called()

    # Recreate only the declared Python attachments, then load them in an
    # isolated interpreter without this checkout on its import path.
    for call in image.add_local_file.call_args_list:
        source, destination = call.args
        if source.endswith(".py"):
            target = tmp_path / destination.removeprefix("/root/")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    code = '''import runpy, sys, types
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
# Import-only stand-ins: no model, dataset or cloud operation is possible.
for name, attributes in {
    "torch": [],
    "datasets": ["Dataset", "get_dataset_config_names", "load_dataset"],
    "peft": ["LoraConfig", "PeftModel", "get_peft_model"],
    "transformers": ["AutoModelForCausalLM", "AutoTokenizer", "Trainer", "TrainingArguments", "default_data_collator"],
}.items():
    module = types.ModuleType(name)
    for attribute in attributes:
        setattr(module, attribute, None)
    sys.modules[name] = module
from cloud.modal import artifacts, processes
for script in sys.argv[2:]:
    assert callable(runpy.run_path(str(root / script))["main"])
'''
    scripts = [remote.removeprefix("/root/")]
    if job == "eos_tune":
        scripts.append("probe_terminate.py")
    result = subprocess.run([sys.executable, "-I", "-c", code, str(tmp_path), *scripts],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("job,trainer,remote", JOBS)
def test_modal_retrieval_keeps_each_jobs_file_selection_and_marker(job, trainer, remote, modal_job, tmp_path):
    namespace, _, volume = modal_job(job)
    entry = namespace["smoke"] if job == "smoke" else namespace["_retrieve"]
    entry.__globals__["REPO_ROOT"] = tmp_path
    names = ["adapter_config.json", "adapter_model.safetensors"]
    if job == "eos_tune":
        names += ["train.log", "probe-base.txt", "probe-tuned.txt"]
    data = b"test artifact"
    manifest = {"status": "complete", "volume_path": "/run", "files": [
        {"name": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} for name in names
    ]}
    volume.read_file.side_effect = lambda path: iter([data[:3], data[3:]])
    if job == "smoke":
        namespace["train_smoke"].remote = lambda: manifest
        invoke = entry
        subdir = "smoke"
    else:
        invoke = lambda: entry(manifest, "test")
        subdir = "test"
    output = tmp_path / "adapters" / subdir
    # A failed transfer must not publish the final filename or completion marker.
    volume.read_file.side_effect = lambda path: iter([b"wrong"])
    with pytest.raises(RuntimeError, match="differs from manifest"):
        invoke()
    assert not (output / names[0]).exists()
    assert not (output / "_COMPLETE.json").exists()
    volume.read_file.reset_mock()
    volume.read_file.side_effect = lambda path: iter([data[:3], data[3:]])
    invoke()
    prefix = namespace["REMOTE_DIR"] if job == "smoke" else "/run"
    assert [call.args[0] for call in volume.read_file.call_args_list] == [f"{prefix}/{name}" for name in names]
    assert all((output / name).read_bytes() == data for name in names)
    assert (output / "_COMPLETE.json").is_file()
    volume.read_file.reset_mock()
    invoke()
    volume.read_file.assert_not_called()


@pytest.fixture
def modal_runtime(modal_job, monkeypatch, tmp_path):
    """Real job orchestration with only training/model/provider work replaced."""
    import contextlib
    import json
    from types import SimpleNamespace
    from cloud.modal import artifacts

    original_hash = artifacts.hash_file
    def local_path(path):
        text = Path(path).as_posix()
        if text.startswith("/root/") or text.startswith("/vault/"):
            return tmp_path / text.lstrip("/")
        return Path(path)
    monkeypatch.setattr(artifacts, "hash_file", lambda path: original_hash(local_path(path)))

    class Encoding(dict):
        def to(self, device): return self
    class Tokenizer:
        pad_token = None
        eos_token = "</s>"
        def __call__(self, *args, **kwargs):
            return Encoding(input_ids=SimpleNamespace(shape=(1, 2)))
        def decode(self, *args, **kwargs): return "translation"
    class Model:
        def to(self, device): return self
        def eval(self): pass
        def generate(self, **kwargs): return [[1, 2, 3]]
        def load_adapter(self, *args, **kwargs): pass
        def add_weighted_adapter(self, *args, **kwargs): pass
        def set_adapter(self, *args): pass
        def merge_and_unload(self): return self
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="fake", inference_mode=contextlib.nullcontext))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=lambda *a, **k: Model())))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: Model())))

    def load(job):
        namespace, image, volume = modal_job(job)
        entry = namespace["train_smoke" if job == "smoke" else "tune_run" if job == "eos_tune" else "train_run"]
        scope = entry.__globals__
        scope["Path"] = lambda path: tmp_path / "vault" if str(path) == "/vault" else Path(path)
        for call in image.add_local_file.call_args_list:
            source, target = call.args
            destination = local_path(target)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.endswith(".py"):
                shutil.copyfile(source, destination)
            else:
                destination.write_bytes(b"input adapter")
        if job == "task":
            for lang in scope["LANGS"]:
                for name in ("adapter_config.json", "adapter_model.safetensors"):
                    path = local_path(f"/vault{scope['_box_dir'](lang, 'smoke')}/{name}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"box input")
        calls = []
        def train(command):
            calls.append(command.copy())
            if "--output" in command:
                output = Path(command[command.index("--output") + 1])
                (output / "adapter_config.json").write_text("{}")
                (output / "adapter_model.safetensors").write_bytes(b"trained")
            return "{'train_loss': 0.5}", "", 0
        scope["run_streaming"] = train
        def buffered(command, **kwargs):
            out, err, rc = train(command)
            return SimpleNamespace(stdout=out, stderr=err, returncode=rc)
        scope["subprocess"] = SimpleNamespace(run=buffered)
        scope["pip_freeze"] = lambda: {}
        args = {"box": ["test", "ro", 500, 20, 8, 2.0, "head"],
                "task": ["test", "smoke", 500, 20, 8, 2.0],
                "monolith": ["test", 500, 20, 8, 2.0],
                "eos_tune": ["test", 2, 5e-5], "smoke": []}[job]
        return SimpleNamespace(entry=entry, scope=scope, args=args, calls=calls,
                               volume=volume, local_path=local_path)
    return load


@pytest.mark.parametrize("job,trainer,remote", JOBS)
def test_modal_request_reuse_and_rejection_are_read_only(job, trainer, remote, modal_runtime):
    import json
    runtime = modal_runtime(job)
    manifest = runtime.entry(*runtime.args)
    command = runtime.calls[0]
    output = Path(command[command.index("--output") + 1])
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    calls = len(runtime.calls)
    assert manifest["request"]["config"]["command"] == command[1:]
    assert len(manifest["request"]["sha256"]) == 64
    assert runtime.entry(*runtime.args) == manifest
    assert len(runtime.calls) == calls
    runtime.volume.commit.assert_called_once()

    runtime.scope["BASE_MODEL"] = "different/requested-model"
    with pytest.raises(RuntimeError, match="identity differs"):
        runtime.entry(*runtime.args)
    runtime.scope["BASE_MODEL"] = manifest["base_model"]
    if runtime.args:
        changed = runtime.args.copy()
        index = {"box": 3, "task": 3, "monolith": 2, "eos_tune": 1}[job]
        changed[index] += 1
        with pytest.raises(RuntimeError, match="identity differs"):
            runtime.entry(*changed)
    else:
        runtime.scope["SMOKE_MAX_STEPS"] += 1
        with pytest.raises(RuntimeError, match="identity differs"):
            runtime.entry()
        runtime.scope["SMOKE_MAX_STEPS"] -= 1
    for remote_source in (remote, "/root/job_definition.py"):
        source = runtime.local_path(remote_source)
        original = source.read_bytes()
        source.write_bytes(original + b"\n# changed defaults\n")
        with pytest.raises(RuntimeError, match="identity differs"):
            runtime.entry(*runtime.args)
        source.write_bytes(original)
    inputs = [item for item in manifest["request"]["config"]["files"] if item["path"].endswith((".json", ".safetensors"))]
    if inputs:
        path = runtime.local_path(next(item["path"] for item in inputs if item["path"].endswith(".safetensors")))
        original = path.read_bytes()
        path.write_bytes(b"changed adapter at the same path")
        with pytest.raises(RuntimeError, match="identity differs"):
            runtime.entry(*runtime.args)
        path.write_bytes(original)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    legacy = dict(manifest)
    del legacy["request"]
    marker = output / "_COMPLETE.json"
    marker.write_text(json.dumps(legacy))
    legacy_bytes = marker.read_bytes()
    with pytest.raises(RuntimeError, match="identity differs or is missing"):
        runtime.entry(*runtime.args)
    assert marker.read_bytes() == legacy_bytes
    assert len(runtime.calls) == calls
    runtime.volume.commit.assert_called_once()


def test_monolith_resume_requires_matching_request(modal_runtime):
    import json
    runtime = modal_runtime("monolith")
    result = runtime.entry(*runtime.args)
    command = runtime.calls[0]
    output = Path(command[command.index("--output") + 1])
    (output / "_COMPLETE.json").unlink()
    checkpoint = output / "checkpoint-10"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"checkpoint")
    changed = runtime.args.copy()
    changed[3] = 16
    with pytest.raises(RuntimeError, match="Incomplete run identity"):
        runtime.entry(*changed)
    assert len(runtime.calls) == 1
    resumed = runtime.entry(*runtime.args)
    assert runtime.calls[-1][-2:] == ["--resume_from", str(checkpoint)]
    assert resumed["request"] == result["request"]
    assert json.loads((output / "_REQUEST.json").read_text()) == result["request"]
