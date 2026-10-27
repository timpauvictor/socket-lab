"""Local experiment boundaries. Provider/GPU operations are replaced, never invoked."""
import argparse
import contextlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cloud.modal.artifacts import hash_file, write_manifest
from cloud.vast import task_eos_artifacts as artifacts
from cloud.vast import task_eos_experiment as experiment
from evaluation import eval_task_eos as evaluation
from training import task_model


def adapter(path, rank=32):
    path.mkdir(parents=True)
    config = {"base_model_name_or_path": artifacts.MODEL, "r": rank, "lora_alpha": rank * 2,
              "lora_dropout": 0.05, "target_modules": sorted(task_model.TARGETS),
              "peft_type": "LORA", "task_type": "CAUSAL_LM", "bias": "none"}
    (path / "adapter_config.json").write_text(json.dumps(config))
    (path / "adapter_model.safetensors").write_bytes(b"original task weights")
    return path


def options(tmp_path, **changes):
    values = dict(repo=tmp_path / "repo", output=tmp_path / "run", run_id="fresh-eos",
                  destination="owner/fresh-eos", model_revision="a" * 40, dataset_revision="b" * 40,
                  provenance=tmp_path / "provenance.json", python=sys.executable, full=True,
                  dry_run=False, approved_live=True, max_seconds=600, max_cost=1, hourly_rate=1)
    values.update(changes)
    return argparse.Namespace(**values)


def test_command_matrix_independent_sources_budgets_and_composition(tmp_path):
    args = options(tmp_path)
    stages = experiment.stage_commands(args)
    training = [(n, c) for n, c in stages if "-train-" in n]
    assert [n for n, _ in training] == ["smoke-train-B", "smoke-train-C", "full-train-B", "full-train-C"]
    assert [c[c.index("--total_steps") + 1] for _, c in training] == ["9", "9", "1000", "1000"]
    assert task_model.allocate_steps(1000, 3) == [333, 333, 334]
    assert task_model.allocate_steps(9, 3) == [3, 3, 3]
    assert {c[c.index("--tune_from") + 1] for _, c in training} == {str(args.output / "bundle/inputs/adapters/task-full")}
    for name, command in training:
        assert ("--eos_stop" in command) == name.endswith("C")
        assert "--paired_control" in command and "--compile" not in command
        assert command[command.index("--pairs") + 1] == "ro-en,fr-en,de-en"
    assert len([n for n, _ in stages if n.startswith("full-eval-")]) == 12
    assert task_model.task_source(0, "original", "new") == "original"
    assert task_model.task_source(1, "original", "new") == "new"
    with pytest.raises(ValueError):
        task_model.allocate_steps(2, 3)


def test_dry_run_never_touches_external_or_write_boundaries(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("dry run crossed a side-effect boundary")
    for target in ("Popen", "run"):
        monkeypatch.setattr(experiment.subprocess, target, forbidden)
    monkeypatch.setattr(experiment, "container_credentials", forbidden)
    monkeypatch.setattr(experiment, "stop_self", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(experiment, "validate_live", forbidden)
    import socket
    import urllib.request
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    assert experiment.main(["run", "--output", str(tmp_path / "never"), "--run-id", "new",
                            "--destination", "owner/new", "--full", "--dry-run"]) == 0
    assert "full-eval-D-de-en" in capsys.readouterr().out


def test_model_owner_composes_once_and_only_task_can_train(tmp_path, monkeypatch):
    calls = []
    params = [("base.weight", SimpleNamespace(requires_grad=True)),
              ("q.lora_A.default.weight", SimpleNamespace(requires_grad=False)),
              ("q.lora_B.default.weight", SimpleNamespace(requires_grad=False)),
              ("q.lora_A.bg0.weight", SimpleNamespace(requires_grad=True))]
    class Model:
        def named_parameters(self): return iter(params)
        def load_adapter(self, path, **kwargs): calls.append(("box1", path, kwargs))
        def add_weighted_adapter(self, names, weights, **kwargs): calls.append(("combine", names, weights, kwargs))
        def set_adapter(self, name): calls.append(("activate", name))
        def merge_and_unload(self): calls.append(("merge",)); return self
    model = Model()
    def load(base, path, **kwargs):
        calls.append(("load", path, kwargs))
        return model
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="BF16"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(
        from_pretrained=lambda *a, **kw: model)))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(LoraConfig=None,
        PeftModel=SimpleNamespace(from_pretrained=load), get_peft_model=lambda *a: pytest.fail("fresh adapter")))
    boxes = tuple(str(adapter(tmp_path / name, 16)) for name in ("ro", "en"))
    task = str(adapter(tmp_path / "task"))
    for trainable in (True, False):
        calls.clear()
        assert task_model.build_task_model(artifacts.MODEL, revision="a" * 40, boxes=boxes,
                                          task_adapter=task, trainable=trainable) is model
        assert calls[2] == ("combine", ["bg0", "bg1"], [1.0, 1.0],
                            {"adapter_name": "combo", "combination_type": "linear"})
        assert calls[4] == ("merge",)
        assert calls[-1] == ("load", task, {"is_trainable": trainable})
        assert [p.requires_grad for _, p in params] == [False, trainable, trainable, False]
    with pytest.raises(ValueError, match="rank"):
        task_model.validate_adapter(task, artifacts.MODEL, 16)
    (Path(task) / "adapter_model.safetensors").unlink()
    with pytest.raises(ValueError, match="Missing"):
        task_model.build_task_model(artifacts.MODEL, task_adapter=task, trainable=True)


@pytest.mark.parametrize("ids,reason", [([7, 999], "eos"), ([7] * 128, "token_cap"), ([7] * 127 + [999], "eos")])
def test_finish_reason_is_observed_from_tokens_even_at_cap(ids, reason):
    result = evaluation.token_evidence(ids, [999], 128)
    assert result["finish_reason"] == reason
    assert result["generated_tokens"] == len(ids)
    if reason == "eos":
        assert result["eos_positions"] == [len(ids) - 1]


@pytest.mark.parametrize("ids", [[], [7], [999, 7], [7] * 129])
def test_unknown_or_invalid_termination_is_not_success(ids):
    with pytest.raises(ValueError):
        evaluation.token_evidence(ids, [999], 128)


@pytest.fixture
def evaluation_runtime(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    for name in ("box-ro", "box-en", "box-de"):
        adapter(inputs / "adapters" / name, 16)
    task = adapter(inputs / "adapters/task-full")
    pairs = tmp_path / "pairs.jsonl"
    with pairs.open("w") as stream:
        stream.write(json.dumps({"_meta": True, "src": "en", "tgt": "ro", "split": "devtest"}) + "\n")
        for i in range(1012):
            stream.write(json.dumps({"id": str(i), "src": "hello", "ref": "salut", "hyp": "NEVER MODEL INPUT"}) + "\n")
    calls = []
    class Tokens:
        shape = (1, 2)
    class Encoding(dict):
        def to(self, device): return self
    class Output:
        def __getitem__(self, key): return SimpleNamespace(tolist=lambda: [7, 8, 999])
    class Tokenizer:
        eos_token_id = pad_token_id = 999
        def __call__(self, prompt, **kwargs):
            assert "NEVER MODEL INPUT" not in prompt
            assert prompt.endswith("Romanian:")
            return Encoding(input_ids=Tokens())
        def decode(self, ids, **kwargs): return "salut\n" + "word " * 30
    class Model:
        def to(self, device): return self
        def eval(self): calls.append("eval")
        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False and kwargs["max_new_tokens"] == 128
            assert kwargs["stop_strings"] is None
            calls.append("generate")
            return Output()
    def build(*args, **kwargs):
        calls.append(kwargs)
        return Model()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *a, **k: Tokenizer())))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=contextlib.nullcontext))
    monkeypatch.setattr(evaluation, "build_task_model", build)
    args = argparse.Namespace(inputs=inputs, adapter=task, pairs=pairs, output=tmp_path / "new.jsonl",
        model=artifacts.MODEL, revision="a" * 40, arm="A", src="en", tgt="ro", run_id="new",
        limit=2, resume=False)
    return args, calls


def test_offline_evaluation_full_owner_path_resume_and_failures(evaluation_runtime, monkeypatch):
    args, calls = evaluation_runtime
    assert evaluation.evaluate(args) == 0
    rows = list(evaluation.iter_records(args.output))
    assert rows[0]["hyp"] == "salut" and rows[0]["looped"] is True
    assert rows[0]["raw"].startswith("salut\n") and rows[0]["finish_reason"] == "eos"
    assert calls[0]["boxes"] == (str(args.inputs / "adapters/box-ro"), str(args.inputs / "adapters/box-en"))
    before = args.output.read_bytes()
    with pytest.raises(ValueError, match="Fresh"):
        evaluation.evaluate(args)
    args.resume = True
    calls.clear()
    assert evaluation.evaluate(args) == 0 and not calls
    assert args.output.read_bytes() == before
    args.revision = "b" * 40
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        evaluation.evaluate(args)
    assert args.output.read_bytes() == before
    args.revision = "a" * 40
    args.output = args.output.with_name("failure.jsonl")
    monkeypatch.setattr(evaluation, "token_evidence", lambda *a: (_ for _ in ()).throw(ValueError("bad generation")))
    with pytest.raises(RuntimeError, match="partial file retained"):
        evaluation.evaluate(args)
    row = list(evaluation.iter_records(args.output, fields=("id", "src", "ref")))[0]
    assert row["error"] == {"type": "ValueError"} and "hyp" not in row and "looped" not in row
    summary = evaluation.summarize(args.output)
    assert summary["failed"] == 1 and summary["missing"] == 1 and summary["complete"] is False
    assert summary["chrf"] is summary["repetition_fraction"] is summary["eos_stop_rate"] is None


def test_monolith_evaluates_without_boxes(evaluation_runtime):
    args, calls = evaluation_runtime
    args.arm = "D"
    assert evaluation.evaluate(args) == 0
    assert calls[0]["boxes"] is None


def test_trainer_loads_original_first_then_carries_task_for_exact_total(tmp_path, monkeypatch):
    class Model:
        def print_trainable_parameters(self): pass
        def save_pretrained(self, output):
            Path(output).mkdir(parents=True, exist_ok=True)
    configurations = []
    class Trainer:
        def __init__(self, model, args, **kwargs):
            configurations.append(args)
            self.state = SimpleNamespace(global_step=args.max_steps, log_history=[{"loss": 1.0}])
        def train(self): pass
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)))
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=None, get_dataset_config_names=None, load_dataset=None))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(LoraConfig=None, PeftModel=None, get_peft_model=None))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModelForCausalLM=None,
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(pad_token="pad")),
        Trainer=Trainer, TrainingArguments=lambda **kw: SimpleNamespace(**kw),
        default_data_collator=None, TrainerCallback=object))
    monkeypatch.delitem(sys.modules, "training.train_task", raising=False)
    training = importlib.import_module("training.train_task")
    monkeypatch.setattr(training, "tokenize_batch", lambda *a, **k: [1, 2])
    calls = []
    monkeypatch.setattr(training, "build_task_model", lambda *a, **k: calls.append(k) or Model())
    data = tmp_path / "data"
    data.mkdir()
    for pair in artifacts.PAIRS:
        (data / f"{pair}.json").write_text('[["hello", "salut"]]')
    output = tmp_path / "new"
    original = tmp_path / "original"
    box = tmp_path / "box"
    box.mkdir()
    argv = ["train_task.py", "--model", artifacts.MODEL, "--output", str(output),
            "--tune_from", str(original), "--total_steps", "1000", "--revision", "a" * 40,
            "--paired_control", "--prepared_data", str(data)]
    for lang in ("ro", "en", "fr", "de"):
        argv += [f"--box_{lang}", str(box)]
    monkeypatch.setattr(sys, "argv", argv)
    training.main()
    assert [c["task_adapter"] for c in calls] == [str(original), str(output), str(output)]
    assert [c.max_steps for c in configurations] == [333, 333, 334]
    assert len({c.output_dir for c in configurations}) == 3
    assert all(c.warmup_ratio == 0.05 and c.lr_scheduler_type == "cosine" for c in configurations)
    assert all(c["trainable"] is True for c in calls)
    before = len(calls)
    with pytest.raises(SystemExit):
        training.main()
    assert len(calls) == before


def test_bundle_retains_nested_partial_data_and_rejects_tampering(tmp_path):
    root = tmp_path / "bundle"
    (root / "full/B/checkpoint-333").mkdir(parents=True)
    (root / "full/B/checkpoint-333/weights").write_bytes(b"partial adapter")
    (root / "completion.json").write_text('{"status":"incomplete"}')
    artifacts.seal_bundle(root)
    artifacts.verify_bundle(root)
    (root / "full/B/checkpoint-333/weights").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        artifacts.verify_bundle(root)


def test_freeze_selects_data_once_copies_originals_and_binds_sources(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    provenance = {}
    for name in artifacts.ADAPTERS:
        path = adapter(repo / "adapters" / name, 16 if name.startswith("box-") else 32)
        provenance[name] = {"revision": "47a09863e61d03e0ef5bbae12cd36516ac770446" if name == "eos-tune-001" else "a" * 40,
            "source": "inspected source", "evidence": "source checksum receipt", "sha256": {
                n: hash_file(path / n)[0] for n in ("adapter_config.json", "adapter_model.safetensors")}}
    # Original language boxes came from Modal and have no Hub commit.
    manifest = {"status": "complete", "base_model": artifacts.MODEL, "rank": 16, "lang": "ro",
                "run_id": "qwen06b-box-ro-full-001", "volume": "historical-vault", "volume_path": "/box-ro/full",
                "files": [{"name": name, "sha256": digest} for name, digest in provenance["box-ro"]["sha256"].items()]}
    marker = repo / "adapters/box-ro/_COMPLETE.json"
    marker.write_text(json.dumps(manifest))
    provenance["box-ro"].update(source_type="modal_manifest", source="modal-volume:historical-vault:/box-ro/full",
                               run_id=manifest["run_id"], manifest_sha256=hash_file(marker)[0])
    del provenance["box-ro"]["revision"]
    artifacts.validate_provenance_entry(repo, "box-ro", provenance["box-ro"])
    for field, value in (("manifest_sha256", "0" * 64), ("run_id", "smoke")):
        with pytest.raises(ValueError, match="provenance|manifest"):
            artifacts.validate_provenance_entry(repo, "box-ro", dict(provenance["box-ro"], **{field: value}))
    for source in artifacts.SOURCES:
        path = repo / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen code")
    for direction in artifacts.DIRECTIONS:
        path = repo / f"logs/eval-{direction}-devtest.jsonl"
        path.parent.mkdir(exist_ok=True)
        src, tgt = direction.split("-")
        with path.open("w") as stream:
            stream.write(json.dumps({"_meta": True, "src": src, "tgt": tgt, "split": "devtest"}) + "\n")
            for i in range(1012):
                stream.write(json.dumps({"id": str(i), "src": "hello", "ref": "salut"}) + "\n")
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(provenance))
    loaded = []
    def load(name, config, **kwargs):
        loaded.append((config, kwargs))
        return [{"translation": {lang: "hello" for lang in ("ro", "en", "fr", "de")}}]
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        get_dataset_config_names=lambda *a, **k: ["ro-en", "fr-en", "de-en"], load_dataset=load))
    monkeypatch.setattr(artifacts, "runtime_metadata", lambda: {"packages": "test"})
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": [list(range(min(len(t), kwargs["max_length"]))) for t in texts],
                    "offset_mapping": [[(i, i + 1) for i in range(min(len(t), kwargs["max_length"]))] for t in texts]}
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda name, revision: Tokenizer())))
    output = tmp_path / "frozen"
    artifacts.freeze_inputs(repo, output, "a" * 40, "b" * 40, provenance_path)
    assert len(loaded) == 3 and all(k["revision"] == "b" * 40 for _, k in loaded)
    assert json.loads((output / "data/ro-en.json").read_text()) == [["hello", "hello"]]
    assert json.loads((output / "data/ro-en.selected.json").read_text()) == [["hello", "hello"]]
    assert json.loads((output / "data/ro-en.excluded.json").read_text()) == []
    selections = json.loads((output / "identity.json").read_text())["selections"]
    assert selections["ro-en"]["raw_selected_pairs"] == selections["ro-en"]["selected_pairs"] == 1
    assert selections["ro-en"]["excluded_pairs"] == 0
    artifacts.check_frozen(output, repo)
    (repo / "adapters/task-full/adapter_model.safetensors").write_bytes(b"changed original")
    assert (output / "adapters/task-full/adapter_model.safetensors").read_bytes() == b"original task weights"
    (repo / artifacts.SOURCES[0]).write_text("# drift")
    with pytest.raises(RuntimeError, match="code drift"):
        artifacts.check_frozen(output, repo)
    with pytest.raises(ValueError, match="fresh"):
        artifacts.freeze_inputs(repo, output, "a" * 40, "b" * 40, provenance_path)


def test_bundle_upload_checks_downloaded_hash_at_uploaded_commit(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "scores.json").write_text("[]")
    artifacts.seal_bundle(root)
    calls = []
    class API:
        def create_repo(self, *a, **kw):
            assert kw == {"repo_type": "model", "private": True, "exist_ok": False}
        def upload_folder(self, **kw): return SimpleNamespace(oid="a" * 40)
        def list_repo_files(self, *a, **kw): return ["manifest.json", "scores.json"]
    def download(repo, name, revision):
        calls.append((name, revision))
        return str(root / name)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=API, hf_hub_download=download))
    result = artifacts.upload_bundle(root, "owner/new")
    assert result["files_verified"] == 2
    assert calls == [("manifest.json", "a" * 40), ("scores.json", "a" * 40)]
    bad = tmp_path / "bad"
    bad.write_bytes(b"wrong bytes")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=API, hf_hub_download=lambda *a, **kw: str(bad)))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        artifacts.upload_bundle(root, "owner/new")


@pytest.mark.parametrize("failed_stage", ["smoke-train-B", "full-eval-C-ro-en", "finalize", None])
def test_supervisor_retains_artifacts_finalizes_and_stops_on_all_outcomes(tmp_path, monkeypatch, failed_stage):
    args = options(tmp_path)
    monkeypatch.setattr(experiment, "validate_live", lambda a: None)
    monkeypatch.setattr(experiment, "check_frozen", lambda *a: None)
    events = []
    class Guard:
        def poll(self): return None
    monkeypatch.setattr(experiment.subprocess, "Popen", lambda *a, **k: Guard())
    monkeypatch.setattr(experiment, "terminate", lambda p: events.append("terminate-guard"))
    monkeypatch.setattr(experiment, "stop_self", lambda: events.append("stop") or True)
    def stage(name, command, run, guard, deadline, progress):
        events.append(name)
        (run / f"{name}.stdout.log").write_text("stage evidence")
        if name == failed_stage:
            raise RuntimeError("observed failure")
    monkeypatch.setattr(experiment, "run_stage", stage)
    code = experiment.run_experiment(args)
    assert code == (0 if failed_stage is None else 1)
    assert "finalize" in events and events.count("stop") == 1
    if failed_stage and failed_stage != "finalize":
        assert events.index("finalize") > events.index(failed_stage)
    assert (args.output / "bundle/stage-logs/smoke-train-B.stdout.log").exists()
    completion = json.loads((args.output / "bundle/completion.json").read_text())
    assert completion["status"] == ("complete" if failed_stage in (None, "finalize") else "incomplete")
    closeout = json.loads((args.output / "closeout.json").read_text())
    assert closeout["upload_verified"] == (failed_stage != "finalize") and closeout["stop_acknowledged"]


def test_failed_stop_keeps_independent_guard_alive(tmp_path, monkeypatch):
    args = options(tmp_path)
    monkeypatch.setattr(experiment, "validate_live", lambda a: None)
    monkeypatch.setattr(experiment, "check_frozen", lambda *a: None)
    monkeypatch.setattr(experiment.subprocess, "Popen", lambda *a, **k: SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr(experiment, "run_stage", lambda *a: None)
    monkeypatch.setattr(experiment, "stop_self", lambda: False)
    monkeypatch.setattr(experiment, "terminate", lambda p: pytest.fail("guard removed despite failed stop"))
    assert experiment.run_experiment(args) == 1
    assert json.loads((args.output / "closeout.json").read_text())["stop_acknowledged"] is False


def test_guard_launch_failure_still_attempts_stop(tmp_path, monkeypatch):
    args = options(tmp_path)
    monkeypatch.setattr(experiment, "validate_live", lambda a: None)
    def fail(*args, **kwargs): raise OSError("launch failed")
    monkeypatch.setattr(experiment.subprocess, "Popen", fail)
    stops = []
    monkeypatch.setattr(experiment, "stop_self", lambda: stops.append(True) or True)
    assert experiment.run_experiment(args) == 1
    assert stops
    assert (args.output / "bundle/completion.json").exists()


def test_independent_watchdog_enforces_deadline(tmp_path, monkeypatch):
    progress = tmp_path / "progress.log"
    progress.touch()
    from cloud.vast import lifecycle
    monkeypatch.setattr(lifecycle, "container_credentials", lambda: ("123", "private"))
    monkeypatch.setattr(lifecycle.time, "time", lambda: 100)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda n: None)
    stops = []
    monkeypatch.setattr(lifecycle, "stop_self", lambda: stops.append(True) or True)
    assert experiment.watchdog(progress, 90) == 0 and stops == [True]


def test_stage_deadline_cancels_owned_process(tmp_path, monkeypatch):
    process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(experiment.subprocess, "Popen", lambda *a, **k: process)
    cancelled = []
    monkeypatch.setattr(experiment, "terminate", lambda p: cancelled.append(p))
    with pytest.raises(RuntimeError, match="time exhausted"):
        experiment.run_stage("train", ["unused"], tmp_path, SimpleNamespace(poll=lambda: None), 0, tmp_path / "progress")
    assert cancelled == [process]
