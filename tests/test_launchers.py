from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_eval
import run_train
import serve


def test_eval_score_only_command_and_inference_dependencies(tmp_path, monkeypatch):
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation/eval_llama.py").touch()
    monkeypatch.setattr(run_eval, "resolve_executable", lambda _: "uv path")
    score = run_eval.build_parser().parse_args(["--repo-root", str(tmp_path), "--score-only", "--output", "out file"])
    cmd = run_eval.command_for(score)
    assert cmd[:7] == ["uv path", "run", "--no-project", "--with", "sacrebleu==2.6.0", "--", "python"]
    assert "datasets" not in cmd and "requests==2.34.2" not in cmd
    assert "--score_only" in cmd and "out file" in cmd
    infer = run_eval.build_parser().parse_args(["--repo-root", str(tmp_path), "--run-id", "r"])
    assert "datasets" in run_eval.command_for(infer)
    assert "requests==2.34.2" in run_eval.command_for(infer)
    assert run_eval.command_for(infer)[-6:] == ["--src", "ro", "--tgt", "en", "--run_id", "r"]


def test_train_mappings_validation_and_logged_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(run_train, "resolve_executable", lambda _: "modal path")
    box = run_train.build_parser().parse_args(["--job", "box", "--phase", "full", "--lang", "all"])
    assert run_train.command_for(box)[0] == ["modal path", "run", "cloud/modal/train_box.py::app.full_all"]
    smoke = run_train.build_parser().parse_args(["--job", "smoke", "--phase", "smoke"])
    assert run_train.command_for(smoke)[1] == "logs/train-smoke-console.log"
    with pytest.raises(ValueError):
        run_train.command_for(run_train.build_parser().parse_args(["--job", "smoke", "--phase", "full"]))

    records = {}
    def fake(command, *, cwd, env, log_path):
        records.update(command=command, cwd=cwd, env=env, log=log_path)
        return 17
    monkeypatch.setattr(run_train, "run_logged", fake)
    assert run_train.main(["--job", "task", "--phase", "smoke", "--repo-root", str(tmp_path)]) == 17
    assert records["cwd"] == tmp_path and records["env"]["PYTHONIOENCODING"] == "utf-8"


def test_dry_runs_and_serve_lora_validation_do_not_spawn_or_write(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_eval, "run_foreground", lambda *a, **k: pytest.fail("spawn"))
    monkeypatch.setattr(run_train, "run_logged", lambda *a, **k: pytest.fail("spawn"))
    monkeypatch.setattr(serve, "run_foreground", lambda *a, **k: pytest.fail("spawn"))
    assert run_train.main(["--job", "smoke", "--phase", "smoke", "--repo-root", str(tmp_path), "--dry-run"]) == 0
    assert not (tmp_path / "logs").exists()
    assert run_eval.main(["--score-only", "--output", str(tmp_path / "x"), "--dry-run"]) == 0
    assert not (tmp_path / "x").exists()
    model = tmp_path / "model file.gguf"
    model.touch()
    adapter = tmp_path / "adapter one.gguf"
    adapter.touch()
    monkeypatch.setattr(serve, "resolve_executable", lambda _: "server path")
    args = serve.build_parser().parse_args(["--model", str(model), "--lora", str(adapter), "--lora", str(adapter)])
    command = serve.command_for(args)
    assert command[0] == "server path" and command[-2:] == ["--lora", f"{adapter},{adapter}"]
    assert serve.main(["--model", str(model), "--lora", "missing.gguf", "--dry-run"]) == 1
    assert capsys.readouterr().err


def test_logged_helper_streams_utf8_and_preserves_exit(tmp_path):
    log = tmp_path / "logs/out.log"
    code = "import sys; print('héllo'); print('diagnostic', file=sys.stderr); sys.exit(17)"
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    result = __import__("tools.processes", fromlist=["run_logged"]).run_logged(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, log_path=log)
    assert result == 17
    assert "héllo" in log.read_text(encoding="utf-8") and "diagnostic" in log.read_text(encoding="utf-8")


def test_training_log_admission_and_explicit_fresh_path(monkeypatch, tmp_path):
    from tools import processes
    calls = []
    class Child:
        def __init__(self): self.stdout = io.StringIO("new invocation\n")
        def wait(self): return 17
        def poll(self): return 17
    def spawn(*args, **kwargs):
        calls.append(args)
        return Child()
    monkeypatch.setattr(processes.subprocess, "Popen", spawn)
    monkeypatch.setattr(run_train, "resolve_executable", lambda _: "stub-modal")
    args = ["--job", "task", "--phase", "smoke", "--repo-root", str(tmp_path)]
    default = tmp_path / "logs/train-task-smoke-console.log"
    default.parent.mkdir()
    default.write_bytes(b"banked")
    assert run_train.main(args) == 1
    assert default.read_bytes() == b"banked" and not calls
    for selected in [Path("fresh/relative.log"), tmp_path / "absolute.log"]:
        assert run_train.main([*args, "--log-file", str(selected)]) == 17
        assert (tmp_path / selected).read_text(encoding="utf-8") == "new invocation\n"
    assert len(calls) == 2 and default.read_bytes() == b"banked"
    dry = tmp_path / "dry/new.log"
    assert run_train.main([*args, "--log-file", str(dry), "--dry-run"]) == 0
    assert not dry.parent.exists() and len(calls) == 2

    # A file arriving just before open must not be truncated either.
    racing = tmp_path / "racing.log"
    original_open = Path.open
    def race(path, mode="r", *a, **kw):
        if path == racing:
            with original_open(path, "wb") as stream:
                stream.write(b"arrived first")
        return original_open(path, mode, *a, **kw)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", race)
        assert run_train.main([*args, "--log-file", str(racing)]) == 1
    assert racing.read_bytes() == b"arrived first" and len(calls) == 2
    for name, target in [("linked.log", default), ("dangling.log", tmp_path / "absent")]:
        link = tmp_path / name
        link.symlink_to(target)
        assert run_train.main([*args, "--log-file", str(link)]) == 1
        assert link.is_symlink() and len(calls) == 2
    assert default.read_bytes() == b"banked" and not (tmp_path / "absent").exists()


def test_foreground_launchers_preserve_environment_and_exit(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()
    calls = []
    def execute(command, *, cwd, env):
        calls.append((command, cwd, env))
        return 17
    monkeypatch.setattr(serve, "resolve_executable", lambda _: "server")
    monkeypatch.setattr(serve, "run_foreground", execute)
    before = dict(os.environ)
    assert serve.main(["--model", str(model)]) == 17
    assert calls[-1][1] == Path.cwd()
    assert calls[-1][2]["DISABLE_RTSS_LAYER"] == "1"
    assert calls[-1][0][calls[-1][0].index("--host") + 1] == "127.0.0.1"
    monkeypatch.setattr(run_eval, "resolve_executable", lambda _: "uv")
    monkeypatch.setattr(run_eval, "run_foreground", execute)
    assert run_eval.main(["--score-only", "--output", "saved records.jsonl"]) == 17
    assert calls[-1][1] == Path.cwd()
    assert dict(os.environ) == before


def test_logging_failure_cancels_child_and_bad_log_never_spawns(monkeypatch, tmp_path):
    from tools import processes
    events = []
    class Output:
        def __iter__(self): return iter(["line\n"])
        def close(self): events.append("closed")
    class Child:
        stdout = Output()
        returncode = None
        def poll(self): return self.returncode
        def terminate(self): events.append("terminated")
        def wait(self, timeout=None): self.returncode = -15; return self.returncode
    monkeypatch.setattr(processes.subprocess, "Popen", lambda *a, **k: Child())
    def broken_write(line):
        raise OSError("console closed")
    monkeypatch.setattr(processes, "sys", SimpleNamespace(stdout=SimpleNamespace(write=broken_write, flush=lambda: None)))
    with pytest.raises(OSError, match="console closed"):
        processes.run_logged(["stub"], cwd=tmp_path, env={}, log_path=tmp_path / "out.log")
    assert events == ["terminated", "closed"]
    monkeypatch.setattr(processes.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned before log opened"))
    with pytest.raises(OSError):
        processes.run_logged(["stub"], cwd=tmp_path, env={}, log_path=tmp_path)
