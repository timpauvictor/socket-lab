"""Local tests for shared Modal artifact and process helpers."""
import hashlib
import json
import os
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
    partial, = fresh.glob(".model.*.part")
    assert partial.read_bytes() == data
    retrieve_files(manifest, fresh, read_file, ["model"])
    assert (fresh / "model").read_bytes() == data
    assert partial.read_bytes() == data  # A retry preserves the failed transfer evidence.
    assert list(fresh.glob(".model.*.part")) == [partial]


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


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:stream", "a/b", "a\\b", "..", "NUL.bin", "COM1", "file.", "_COMPLETE.json", "_complete.json"])
def test_retrieval_admits_all_names_before_transferring(tmp_path, name):
    entry = {"name": "first", "sha256": hashlib.sha256(b"good").hexdigest(), "bytes": 4}
    manifest = {"volume_path": "/remote", "files": [entry, dict(entry, name=name)]}
    with pytest.raises(RuntimeError, match="Unsafe artifact"):
        retrieve_files(manifest, tmp_path, lambda _: pytest.fail("read before admission"), ["first", name])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("extra,changes,selected", [
    ("second", {"sha256": "invalid"}, "second"), ("second", {"bytes": True}, "second"),
    ("second", {"bytes": -1}, "second"), ("model", {}, "model"),
    ("MODEL", {}, "MODEL"), ("second", {}, "missing")])
def test_retrieval_rejects_invalid_or_ambiguous_manifest(tmp_path, extra, changes, selected):
    entry = {"name": "model", "sha256": hashlib.sha256(b"good").hexdigest(), "bytes": 4}
    manifest = {"volume_path": "/remote", "files": [entry, dict(entry, name=extra, **changes)]}
    with pytest.raises(RuntimeError, match="Invalid or duplicate|missing from manifest"):
        retrieve_files(manifest, tmp_path, lambda _: pytest.fail("read before admission"), ["model", selected])
    assert not list(tmp_path.iterdir())


def test_retrieval_preserves_linked_partial_and_rejects_destination_link(tmp_path):
    data = b"good"
    outside = tmp_path / "outside"
    outside.write_bytes(b"banked")
    local = tmp_path / "local"
    local.mkdir()
    os.link(outside, local / "model.part")
    entry = {"name": "model", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    manifest = {"volume_path": "/remote", "files": [entry]}
    retrieve_files(manifest, local, lambda _: [data], ["model"])
    assert (local / "model").read_bytes() == data
    assert outside.read_bytes() == (local / "model.part").read_bytes() == b"banked"
    linked = tmp_path / "linked"
    linked.mkdir()
    try:
        (linked / "model").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Host cannot create symlinks: {exc}")
    with pytest.raises(RuntimeError, match="Unsafe local artifact"):
        retrieve_files(manifest, linked, lambda _: pytest.fail("download through link"), ["model"])
    assert outside.read_bytes() == b"banked" and (linked / "model").is_symlink()


def test_environment_capture_keeps_only_recorded_packages(monkeypatch):
    installed = {"torch": "2.12.0+cpu", "peft": "0.21.2", "other": "1"}
    queried = []

    def version(name):
        queried.append(name)
        if name not in installed:
            raise processes.PackageNotFoundError(name)
        return installed[name]

    monkeypatch.setattr(processes, "version", version)
    monkeypatch.setattr(processes.subprocess, "run", lambda *a, **k: pytest.fail("spawned pip"))
    assert processes.package_versions() == {"torch": "2.12.0+cpu", "peft": "0.21.2"}
    assert queried == ["torch", "transformers", "peft", "datasets", "accelerate"]

    def corrupt_metadata(name):
        raise RuntimeError("unreadable metadata")

    monkeypatch.setattr(processes, "version", corrupt_metadata)
    with pytest.raises(RuntimeError, match="unreadable metadata"):
        processes.package_versions()


def test_streaming_output_and_exit_status():
    code = "import sys; print('out'); print('err', file=sys.stderr); sys.exit(7)"
    stdout, stderr, status = run_streaming([sys.executable, "-c", code])
    assert stdout == "out\n"
    assert stderr == "err\n"
    assert status == 7
