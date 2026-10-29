"""Local tests for shared Modal artifact and process helpers."""
import hashlib
import json
import sys

import pytest

from cloud.modal.artifacts import file_manifest, hash_file, retrieve_files, write_manifest
from cloud.modal import processes
from cloud.modal.processes import run_streaming


def test_hash_manifest_and_serialization(tmp_path):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"abc")
    expected = hashlib.sha256(b"abc").hexdigest()
    assert hash_file(source) == (expected, 3)
    manifest = {"files": file_manifest(tmp_path, ["weights.bin"])}
    assert manifest["files"] == [{"name": "weights.bin", "sha256": expected, "bytes": 3}]
    target = tmp_path / "manifest.json"
    write_manifest(target, {"z": 1, "a": 2})
    assert target.read_text(encoding="utf-8") == json.dumps({"z": 1, "a": 2}, indent=2, sort_keys=True) + "\n"
    original = b'{"z":1,"a":2}'
    target.write_bytes(original)
    write_manifest(target, {"a": 2, "z": 1})
    assert target.read_bytes() == original
    with pytest.raises(RuntimeError, match="Conflicting manifest"):
        write_manifest(target, {"a": 3})
    assert target.read_bytes() == original


def test_retrieval_chunks_skip_conflict_and_mismatch(tmp_path):
    data = b"verified data"
    entry = {"name": "model", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    manifest = {"volume_path": "/remote/job", "files": [entry]}
    calls = []
    def read_file(path):
        calls.append(path)
        return [data[:4], data[4:]]
    retrieve_files(manifest, tmp_path, read_file, ["model"])
    assert (tmp_path / "model").read_bytes() == data
    retrieve_files(manifest, tmp_path, read_file, ["model"])
    assert calls == ["/remote/job/model"]
    bad = dict(manifest, files=[dict(entry, bytes=len(data) + 1)])
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    with pytest.raises(RuntimeError, match="differs from manifest"):
        retrieve_files(bad, fresh, read_file, ["model"], remote_path="/chosen")
    assert calls[-1] == "/chosen/model"
    assert not (fresh / "model").exists()
    assert (fresh / "model.part").read_bytes() == data


def test_retrieval_rejects_conflicting_receipt_before_download(tmp_path):
    marker = tmp_path / "_COMPLETE.json"
    original = b'{"run_id":"historical"}'
    marker.write_bytes(original)
    with pytest.raises(RuntimeError, match="Conflicting manifest"):
        retrieve_files({"run_id": "new", "files": []}, tmp_path,
                       lambda _: pytest.fail("download before identity check"), [])
    assert marker.read_bytes() == original
    assert list(tmp_path.iterdir()) == [marker]


def test_retrieval_rejects_conflicting_existing_file(tmp_path):
    data = b"good"
    manifest = {"files": [{"name": "model", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}]}
    (tmp_path / "model").write_bytes(b"bad")
    with pytest.raises(RuntimeError, match="Conflicting local"):
        retrieve_files(manifest, tmp_path, lambda _: pytest.fail("download should be skipped"), ["model"])


def test_environment_capture_keeps_only_recorded_packages(monkeypatch):
    from types import SimpleNamespace

    def freeze(command, **kwargs):
        assert command == [sys.executable, "-m", "pip", "freeze"]
        assert kwargs == {"capture_output": True, "text": True, "check": True}
        return SimpleNamespace(stdout="torch==2.12.0+cpu\npeft==0.21.2\nother==1\neditable @ file:///ignored\n")
    monkeypatch.setattr(processes.subprocess, "run", freeze)
    assert processes.pip_freeze() == {"torch": "2.12.0+cpu", "peft": "0.21.2"}


def test_streaming_output_and_exit_status():
    code = "import sys; print('out'); print('err', file=sys.stderr); sys.exit(7)"
    stdout, stderr, status = run_streaming([sys.executable, "-c", code])
    assert stdout == "out\n"
    assert stderr == "err\n"
    assert status == 7
