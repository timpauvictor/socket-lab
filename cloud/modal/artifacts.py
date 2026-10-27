"""Shared artifact hashing, manifests, and verified file retrieval."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path


def hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def file_manifest(out_dir: Path, names: Iterable[str]) -> list[dict[str, str | int]]:
    files = []
    for name in names:
        sha, size = hash_file(out_dir / name)
        files.append({"name": name, "sha256": sha, "bytes": size})
    return files


def _require_matching_manifest(path: Path, manifest: dict) -> None:
    if json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError(f"Conflicting manifest already exists: {path}")


def write_manifest(path: Path, manifest: dict) -> None:
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError:
        _require_matching_manifest(path, manifest)
        return  # Preserve even the original formatting of a matching receipt.
    with stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def request_identity(command: list[str], *, sources: Iterable[str] = (),
                     inputs: Iterable[str] = ()) -> dict:
    """Identify requested arguments and declared training sources/input bytes.

    Interpreter location and automatic checkpoint recovery are not recipe
    settings. Remote model/dataset names are declarations, not pinned revisions.
    """
    paths = sorted({command[1], *sources, *inputs})
    files = []
    for name in paths:
        sha, size = hash_file(Path(name))
        files.append({"path": name, "sha256": sha, "bytes": size})
    config = {"version": 1, "command": command[1:], "files": files}
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"config": config, "sha256": hashlib.sha256(encoded).hexdigest()}


def check_run_request(out_dir: Path, run_id: str, request: dict) -> dict | None:
    """Reject stale/legacy completion or recovery state before any mutation."""
    marker = out_dir / "_COMPLETE.json"
    if marker.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if (isinstance(existing, dict) and existing.get("status") == "complete"
                and existing.get("run_id") == run_id and existing.get("request") == request):
            print("Completed request identity matches; reusing, no training ran", flush=True)
            return existing
        raise RuntimeError(f"Completed run identity differs or is missing: {marker}; choose a new run ID")
    if out_dir.exists() and any(out_dir.iterdir()):
        pending = out_dir / "_REQUEST.json"
        if not pending.is_file() or json.loads(pending.read_text(encoding="utf-8")) != request:
            raise RuntimeError(f"Incomplete run identity differs or is missing: {out_dir}; inspect before retry")
    return None


def retrieve_files(manifest: dict, local_dir: Path,
                   read_file: Callable[[str], Iterable[bytes]], names: Iterable[str],
                   *, remote_path: str | None = None) -> None:
    """Verify selected files before replacement; caller publishes completion last."""
    marker = local_dir / "_COMPLETE.json"
    if marker.exists():
        _require_matching_manifest(marker, manifest)
    by_name = {entry["name"]: entry for entry in manifest["files"]}
    for name in names:
        entry = by_name[name]
        local_file = local_dir / name
        if local_file.exists():
            if hash_file(local_file) != (entry["sha256"], entry["bytes"]):
                raise RuntimeError(f"Conflicting local file exists: {local_file}")
            print(f"Existing local file verified, no download: {name}", flush=True)
            continue
        partial = local_dir / (name + ".part")
        digest = hashlib.sha256()
        size = 0
        directory = manifest["volume_path"] if remote_path is None else remote_path
        remote = f"{directory}/{name}"
        with partial.open("wb") as out:
            for chunk in read_file(remote):
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if (digest.hexdigest(), size) != (entry["sha256"], entry["bytes"]):
            raise RuntimeError(f"Downloaded {name} differs from manifest")
        partial.replace(local_file)
        print(f"Downloaded and verified: {local_file}", flush=True)
