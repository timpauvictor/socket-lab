import json
import os
import sys
import types

import pytest

from cloud.vast import lifecycle, provision, recipes, run, upload


def install_credentials(monkeypatch):
    monkeypatch.setenv("CONTAINER_ID", "123")
    monkeypatch.setenv("CONTAINER_API_KEY", "secret")


def adapter_files(output):
    output.mkdir(parents=True)
    (output / "adapter_config.json").write_text("{}")
    (output / "adapter_model.safetensors").write_bytes(b"weights")


def test_finish_success_and_failure_paths_are_verified_and_stop_only(monkeypatch, tmp_path):
    install_credentials(monkeypatch)
    output = tmp_path / "output"
    adapter_files(output)
    log = tmp_path / "logs" / "train.log"
    log.parent.mkdir()
    log.write_text("training output")
    calls = []

    def upload_stage(stage, repo):
        calls.append((repo, {p.name for p in stage.iterdir()}, json.loads((stage / "run-info.json").read_text())))

    stops = []
    monkeypatch.setattr(lifecycle, "upload_and_verify", upload_stage)
    monkeypatch.setattr(lifecycle, "stop_self", lambda: stops.append(True) or True)
    assert lifecycle.finish("job", output, "owner/model", 0, log, keep_running=True) == 0
    assert calls[-1][0] == "owner/model"
    assert {"run-info.json", "train.log", "adapter_config.json", "adapter_model.safetensors"} <= calls[-1][1]
    assert calls[-1][2]["job"] == "job"
    assert calls[-1][2]["train_exit_code"] == 0
    assert calls[-1][2]["completed_at_utc"].endswith("Z")
    assert (tmp_path / "logs" / "epilogue.job.done").exists()
    assert not stops

    assert lifecycle.finish("full", output, "owner/model", 0, log) == 0
    assert stops == [True]
    assert lifecycle.finish("failed", output, "owner/model", 7, log, keep_running=True) == 1
    assert "adapter_model.safetensors" not in calls[-1][1]
    assert stops == [True, True]
    assert calls[-1][2]["train_exit_code"] == 7
    assert (tmp_path / "logs" / "epilogue.failed.done").exists()
    monkeypatch.setattr(lifecycle, "stop_self", lambda: False)
    assert lifecycle.finish("stop-failed", output, "owner/model", 0, log) == 1
    assert not lifecycle.markers(log, "stop-failed")[1].exists()


def test_finish_incomplete_or_upload_failure_stops_without_completion(monkeypatch, tmp_path):
    install_credentials(monkeypatch)
    log = tmp_path / "train.log"
    log.write_text("log")
    output = tmp_path / "empty"
    output.mkdir()
    stops = []
    monkeypatch.setattr(lifecycle, "stop_self", lambda: stops.append(1) or True)
    monkeypatch.setattr(lifecycle, "upload_and_verify", lambda *_: None)
    assert lifecycle.finish("missing", output, "owner/model", 0, log) == 1
    assert not lifecycle.markers(log, "missing")[1].exists()
    monkeypatch.setattr(lifecycle, "upload_and_verify", lambda *_: (_ for _ in ()).throw(OSError("secret")))
    assert lifecycle.finish("broken", output, "owner/model", 0, log) == 1
    assert not lifecycle.markers(log, "broken")[1].exists()
    assert stops == [1, 1]
    recovery = list(tmp_path.glob("upload-broken-*/run-info.json"))
    assert len(recovery) == 1
    assert json.loads(recovery[0].read_text())["job"] == "broken"


def test_finish_dry_run_and_credentials_do_not_mutate(monkeypatch, tmp_path):
    log = tmp_path / "absent" / "train.log"
    monkeypatch.delenv("CONTAINER_ID", raising=False)
    monkeypatch.delenv("CONTAINER_API_KEY", raising=False)
    monkeypatch.setattr(lifecycle, "upload_and_verify", lambda *_: pytest.fail("upload called"))
    monkeypatch.setattr(lifecycle, "stop_self", lambda: pytest.fail("stop called"))
    assert lifecycle.finish("job", tmp_path, "owner/model", 0, log, dry_run=True) == 0
    with pytest.raises(ValueError):
        lifecycle.finish("job", tmp_path, "owner/model", 0, log)
    assert not log.parent.exists()


def test_upload_verifier_and_weight_size(monkeypatch, tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    weights = stage / "adapter_model.safetensors"
    weights.write_bytes(b"abc")
    class Api:
        files = [weights.name]
        size = 3
        def create_repo(self, *args, **kwargs): pass
        def upload_folder(self, **kwargs): pass
        def list_repo_files(self, repo): return self.files
        def hf_hub_download(self, repo, name):
            path = tmp_path / "download"
            path.write_bytes(b"x" * self.size)
            return str(path)
    api = Api()
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=lambda: api))
    lifecycle.upload_and_verify(stage, "owner/model")
    api.files = []
    with pytest.raises(RuntimeError, match="missing"):
        lifecycle.upload_and_verify(stage, "owner/model")
    api.files = [weights.name]
    api.size = 2
    with pytest.raises(RuntimeError, match="size"):
        lifecycle.upload_and_verify(stage, "owner/model")


def test_stop_self_cli_fallback_is_instance_scoped_and_redacted(monkeypatch, capsys):
    install_credentials(monkeypatch)
    commands = []
    monkeypatch.setattr(lifecycle.subprocess, "run", lambda args, **kwargs: commands.append(args) or types.SimpleNamespace(returncode=1))
    requests = []
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self): return b'{"success":true}'
    monkeypatch.setattr(lifecycle.urllib.request, "urlopen", lambda request, **kwargs: requests.append(request) or Response())
    assert lifecycle.stop_self()
    assert commands[0][:4] == ["vastai", "stop", "instance", "123"]
    assert "--destroy" not in commands[0]
    assert requests[0].full_url.endswith("/instances/123/")
    assert json.loads(requests[0].data) == {"state": "stopped"}
    assert requests[0].method == "PUT"
    assert "secret" not in capsys.readouterr().out
    monkeypatch.delenv("CONTAINER_API_KEY")
    with pytest.raises(ValueError):
        lifecycle.stop_self()


def test_stop_reason_boundaries_and_watchdog_retries(monkeypatch, tmp_path):
    log = tmp_path / "train.log"
    log.write_text("complete")
    now = 10000
    os.utime(log, (now, now))
    assert lifecycle.stop_reason(log, "complete", "job", now + 900, now) is None
    assert lifecycle.stop_reason(log, "complete", "job", now + 901, now) == "epilogue never started"
    log.write_text("active")
    os.utime(log, (now, now))
    assert lifecycle.stop_reason(log, "complete", "job", now + 3599, now) is None
    assert lifecycle.stop_reason(log, "complete", "job", now + 3601, now) == "training log is inactive"
    start, done = lifecycle.markers(log, "job")
    start.touch()
    os.utime(start, (now, now))
    assert lifecycle.stop_reason(log, "complete", "job", now + 900, now) is None
    assert lifecycle.stop_reason(log, "complete", "job", now + 901, now) == "epilogue did not finish"
    done.touch()
    assert lifecycle.stop_reason(log, "complete", "job", now + 9999, now) is None
    done.unlink()
    install_credentials(monkeypatch)
    ticks = iter([1, 2, 3])
    monkeypatch.setattr(lifecycle.time, "time", lambda: next(ticks))
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _: None)
    attempts = iter([False, True])
    monkeypatch.setattr(lifecycle, "stop_reason", lambda *args: "stale")
    monkeypatch.setattr(lifecycle, "stop_self", lambda: next(attempts))
    assert lifecycle.watchdog(log, "done", "job") == 0


def test_recipes_keep_historical_arguments_and_resume_rules(tmp_path):
    task = recipes.job_spec("task", tmp_path, "python")
    mono = recipes.job_spec("monolith", tmp_path, "python")
    eos = recipes.job_spec("eos-tune", tmp_path, "python")
    smoke = recipes.job_spec("task-smoke", tmp_path, "python")
    bindings = [task.command[index + 1] for index, value in enumerate(task.command) if value == "--box"]
    assert bindings == [f"{lang}={tmp_path / 'boxes' / ('box-' + lang)}" for lang in ("ro", "en", "fr", "de")]
    assert "--epochs" in task.command and "--compile" in task.command
    assert "--resume_from" in mono.command and "--compile" in mono.command
    assert "--eos_stop" in eos.command and "--tune_from" in eos.command and "--compile" not in eos.command
    assert "--max_steps" in smoke.command and smoke.keep_running
    plain, compiled = recipes.compile_jobs(tmp_path, "python")
    assert "--compile" not in plain.command and "--compile" in compiled.command
    with pytest.raises(ValueError):
        recipes.job_spec("task", tmp_path, "python", resume_from="checkpoint")
    assert recipes.job_spec("task", tmp_path, "python", resume_from="checkpoint", resume_pair="ro-en").command[-4:] == ["--resume_from", "checkpoint", "--resume_pair", "ro-en"]


def test_runner_keeps_guard_through_finalizer_and_handles_launch_failures(monkeypatch, tmp_path):
    install_credentials(monkeypatch)
    job = recipes.Job("job", ["trainer"], tmp_path / "out", tmp_path / "logs" / "train.log", "owner/model", "done")
    monkeypatch.setattr(run, "container_credentials", lambda: ("123", "secret"))
    class Proc:
        def __init__(self, code=None): self.returncode, self.pid = code, 99
        def poll(self): return self.returncode
        def wait(self, timeout=None): return self.returncode
    guard, trainer = Proc(), Proc(0)
    spawned = iter([guard, trainer])
    monkeypatch.setattr(run.subprocess, "Popen", lambda *args, **kwargs: next(spawned))
    events = []
    monkeypatch.setattr(run, "terminate", lambda proc: events.append(("terminate", proc)) if proc.poll() is None else None)
    def finalizer(*args, **kwargs):
        events.append(("finalize", guard.returncode))
        return types.SimpleNamespace(returncode=0 if args[0][6] == "0" else 1)
    monkeypatch.setattr(run.subprocess, "run", finalizer)
    assert run.run_job(job, "python") == 0
    assert events[0] == ("finalize", None)
    assert events[-1] == ("terminate", guard)

    spawned = iter([guard, None])
    def launch(*args, **kwargs):
        item = next(spawned)
        if item is None: raise OSError("launch")
        return item
    monkeypatch.setattr(run.subprocess, "Popen", launch)
    events.clear()
    assert run.run_job(job, "python") != 0
    assert events[0] == ("finalize", None)
    assert ("terminate", guard) not in events

    interrupted = Proc()
    def wait_interrupted(timeout=None):
        raise KeyboardInterrupt
    interrupted.wait = wait_interrupted
    spawned = iter([guard, interrupted])
    monkeypatch.setattr(run.subprocess, "Popen", lambda *a, **k: next(spawned))
    events.clear()
    assert run.run_job(job, "python") == 130
    assert events[0] == ("terminate", interrupted)
    assert ("finalize", None) in events
    assert ("terminate", guard) not in events


def test_upload_commands_and_main_dry_run_and_failure(monkeypatch, tmp_path):
    commands, sources = upload.upload_commands(tmp_path, "example.com", 2222, tmp_path / "checkpoint")
    assert commands[0][0] == "ssh" and all(command[0] == "scp" for command in commands[1:])
    assert tmp_path / "training/translation_data.py" in sources
    assert tmp_path / "cloud/vast/lifecycle.py" in sources
    assert commands[2][-1] == "root@example.com:/workspace/bh/"
    monkeypatch.setattr(sys, "argv", ["upload", "--host", "example.com", "--port", "22", "--repo-root", str(tmp_path), "--dry-run"])
    monkeypatch.setattr(upload.subprocess, "run", lambda *args, **kwargs: pytest.fail("spawned in dry run"))
    assert upload.main() == 0
    for source in sources:
        source.parent.mkdir(parents=True, exist_ok=True)
        if source == tmp_path / "checkpoint":
            source.mkdir()
        else:
            source.touch()
    calls = []
    monkeypatch.setattr(upload.subprocess, "run", lambda command: calls.append(command) or types.SimpleNamespace(returncode=9))
    monkeypatch.setattr(sys, "argv", ["upload", "--host", "example.com", "--port", "22", "--repo-root", str(tmp_path), "--checkpoint", str(tmp_path / "checkpoint")])
    assert upload.main() == 9
    assert len(calls) == 1


def test_provision_dry_run_has_no_import_install_or_write(monkeypatch, tmp_path):
    workspace = tmp_path / "future"
    monkeypatch.setattr(sys, "argv", ["provision", "--workspace", str(workspace), "--dry-run"])
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(provision.subprocess, "run", lambda *args, **kwargs: pytest.fail("installed in dry-run"))
    assert provision.main() == 0
    assert not workspace.exists()


def test_run_cli_dry_run_does_not_require_credentials_or_create_logs(monkeypatch, tmp_path):
    monkeypatch.delenv("CONTAINER_ID", raising=False)
    monkeypatch.delenv("CONTAINER_API_KEY", raising=False)
    monkeypatch.setattr(run, "run_job", lambda *a, **k: pytest.fail("launched"))
    monkeypatch.setattr(run, "stop_self", lambda: pytest.fail("stop"))
    for job in [*recipes.RECIPES, "compile-smoke"]:
        monkeypatch.setattr(sys, "argv", ["run", job, "--workspace", str(tmp_path), "--dry-run"])
        assert run.main() == 0
    assert list(tmp_path.iterdir()) == []


def test_provision_targets_current_interpreter_and_keeps_token_private(monkeypatch, tmp_path, capsys):
    token = "synthetic-private-token"
    monkeypatch.setenv("HF_TOKEN", token)
    torch = types.SimpleNamespace(__version__="test", version=types.SimpleNamespace(cuda="test"),
                                  cuda=types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "fake GPU"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    for name in ("transformers", "peft", "triton"):
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(__version__="test"))
    downloads = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        snapshot_download=lambda *a, **k: downloads.append((a, k))))
    token_path = tmp_path / "private/token"
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", types.SimpleNamespace(HF_TOKEN_PATH=str(token_path)))
    calls = []
    monkeypatch.setattr(provision.subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(provision.shutil, "which", lambda _: "/fake/compiler")
    provision.provision(tmp_path / "workspace")
    assert calls[0][:6] == ["uv", "pip", "install", "--quiet", "--python", sys.executable]
    assert calls[0][6:] == list(provision.PACKAGES)
    assert downloads == [((provision.BASE_MODEL,), {"local_files_only": False})]
    assert token_path.read_text() == token
    assert (tmp_path / "workspace/logs").is_dir()
    assert token not in capsys.readouterr().out


def test_terminate_targets_only_the_owned_process_group(monkeypatch):
    calls = []
    monkeypatch.setattr(run, "os", types.SimpleNamespace(name="posix", killpg=lambda *args: calls.append(args)))
    monkeypatch.setattr(run, "signal", types.SimpleNamespace(SIGTERM=15, SIGKILL=9))
    class Child:
        pid = 123
        count = 0
        def poll(self): return None
        def wait(self, timeout=None):
            self.count += 1
            if self.count == 1:
                raise run.subprocess.TimeoutExpired("owned-child", timeout)
    run.terminate(Child())
    assert calls == [(123, 15), (123, 9)]
