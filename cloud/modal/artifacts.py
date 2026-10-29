"""Shared artifact hashing, manifests, and verified file retrieval."""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
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
    """Reject old or mismatched completion and recovery records before changing files."""
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
    """Download files with simple filenames; save the completion record last.

    Check names and destinations before downloading. Use one writer for each
    directory. These checks cannot protect against another process replacing
    the directory at the same time.
    """
    root = local_dir.resolve(strict=True)
    by_name = {}
    seen_names = set()
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            raise RuntimeError("Invalid artifact manifest entry")
        name = _artifact_name(entry.get("name"))
        if (name.lower() in seen_names or not isinstance(entry.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
                or type(entry.get("bytes")) is not int or entry["bytes"] < 0):
            raise RuntimeError(f"Invalid or duplicate artifact manifest entry: {name}")
        by_name[name] = entry
        seen_names.add(name.lower())
    selected = [_artifact_name(name) for name in names]
    if any(name not in by_name for name in selected):
        raise RuntimeError("Selected artifact missing from manifest")
    marker = root / "_COMPLETE.json"
    if marker.is_symlink():
        raise RuntimeError(f"Unsafe local manifest link: {marker}")
    if marker.exists():
        _require_matching_manifest(marker, manifest)
    pending = []
    for name in selected:
        local_file = root / name
        if local_file.is_symlink() or not local_file.resolve().is_relative_to(root):
            raise RuntimeError(f"Unsafe local artifact path: {local_file}")
        entry = by_name[name]
        if local_file.exists():
            if not local_file.is_file() or hash_file(local_file) != (entry["sha256"], entry["bytes"]):
                raise RuntimeError(f"Conflicting local file exists: {local_file}")
            print(f"Existing local file verified, no download: {name}", flush=True)
        else:
            pending.append(name)
    for name in pending:
        entry = by_name[name]
        local_file = root / name
        digest = hashlib.sha256()
        size = 0
        directory = manifest["volume_path"] if remote_path is None else remote_path
        remote = f"{directory}/{name}"
        out = tempfile.NamedTemporaryFile(mode="wb", dir=root, prefix=f".{name}.", suffix=".part", delete=False)
        partial = Path(out.name)
        try:
            with out:
                for chunk in read_file(remote):
                    out.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if (digest.hexdigest(), size) != (entry["sha256"], entry["bytes"]):
                raise RuntimeError(f"Downloaded {name} differs from manifest")
            partial.replace(local_file)
        finally:
            if partial.exists():
                print(f"Download partial retained for inspection: {partial}", flush=True)
        print(f"Downloaded and verified: {local_file}", flush=True)


def _artifact_name(name: object) -> str:
    # Portable flat names avoid traversal, Windows devices/streams and aliases.
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    if (not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None
            or name.endswith(".") or name.split(".")[0].upper() in reserved
            or name.lower() == "_complete.json"):
        raise RuntimeError(f"Unsafe artifact filename: {name!r}")
    return name
