"""Vast artifact finalization and an independent stop-only watchdog.

Only the instance's own CONTAINER_ID/CONTAINER_API_KEY may authorize a stop.
No destroy operation or account-level credential fallback exists here.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API_URL = "https://console.vast.ai/api/v0"
STALE_SECONDS = 60 * 60
ORPHAN_SECONDS = 15 * 60
CHECK_INTERVAL = 300


def container_credentials() -> tuple[str, str]:
    instance = os.environ.get("CONTAINER_ID", "")
    key = os.environ.get("CONTAINER_API_KEY", "")
    if not instance.isdecimal() or not key.strip():
        raise ValueError("CONTAINER_ID must be numeric and CONTAINER_API_KEY must be set on this instance")
    return instance, key


def markers(log: Path, tag: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("job tag must contain only letters, digits, underscores or hyphens")
    return log.parent / f"epilogue.{tag}.start", log.parent / f"epilogue.{tag}.done"


def stop_self() -> bool:
    instance, key = container_credentials()
    # Capture CLI diagnostics: never print its command line or secret-bearing errors.
    try:
        result = subprocess.run(
            ["vastai", "stop", "instance", instance, "--api-key", key],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            print("Instance stop requested; disk preserved. Confirm in the dashboard.", flush=True)
            return True
    except (OSError, subprocess.TimeoutExpired):
        pass
    print("CLI stop failed; trying the instance-scoped REST endpoint.", flush=True)
    request = urllib.request.Request(
        f"{API_URL}/instances/{instance}/", data=b'{"state":"stopped"}', method="PUT",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
            if not 200 <= response.status < 300:
                return False
            if body:
                payload = json.loads(body)
                if isinstance(payload, dict) and payload.get("success") is False:
                    return False
        print("Instance stop requested via REST; disk preserved.", flush=True)
        return True
    except (OSError, urllib.error.URLError, ValueError):
        print("Stop failed; the instance may still bill. Check the dashboard.", flush=True)
        return False


def upload_and_verify(stage: Path, repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(stage), repo_id=repo, repo_type="model")
    names = set(api.list_repo_files(repo))
    for file in stage.iterdir():
        if file.name not in names:
            raise RuntimeError(f"Uploaded repository is missing {file.name}")
    weights = stage / "adapter_model.safetensors"
    if weights.is_file():
        downloaded = Path(api.hf_hub_download(repo, weights.name))
        if downloaded.stat().st_size != weights.stat().st_size:
            raise RuntimeError("Uploaded adapter weight size does not match")


def finish(tag: str, output: Path, repo: str, train_exit_code: int, log: Path,
           *, keep_running: bool = False, dry_run: bool = False) -> int:
    start, done = markers(log, tag)
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("expected a Hugging Face owner/repository name")
    if dry_run:
        print(f"Would stage {log}, upload to {repo}, verify, and "
              + ("keep running on success." if keep_running else "stop this instance."))
        return 0
    container_credentials()  # Reject missing authority before any upload or marker write.
    uploaded = False
    stopped = False
    stage = None
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        done.unlink(missing_ok=True)
        start.touch()
        stage = Path(tempfile.mkdtemp(prefix=f"upload-{tag}-", dir=log.parent))
        info = {"job": tag, "train_exit_code": train_exit_code,
                "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "hf_repo": repo}
        (stage / "run-info.json").write_text(json.dumps(info) + "\n", encoding="utf-8")
        shutil.copyfile(log, stage / "train.log")
        missing = []
        if train_exit_code == 0:
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                if not (output / name).is_file():
                    missing.append(name)
            if not missing:
                for name in ("adapter_config.json", "adapter_model.safetensors", "manifest.json"):
                    if (output / name).is_file():
                        shutil.copyfile(output / name, stage / name)
        # Even an incomplete successful run should preserve its logs remotely.
        upload_and_verify(stage, repo)
        if missing:
            raise RuntimeError("Required adapter files are missing")
        uploaded = True
        print(f"Upload verified: {repo}", flush=True)
    except Exception as exc:
        # Provider exception text can contain credentials; report the category only.
        print(f"Finalization failed ({type(exc).__name__}); preserving disk and attempting stop.", flush=True)
    finally:
        if uploaded and train_exit_code == 0 and keep_running:
            stopped = True  # Explicit successful smoke workflow: no stop requested.
            print("Smoke artifacts verified; instance intentionally remains running.", flush=True)
        else:
            stopped = stop_self()
        if uploaded and stage is not None:
            try:
                shutil.rmtree(stage)
            except OSError:
                print(f"Verified upload stage could not be cleaned up: {stage}", flush=True)
        elif stage is not None:
            print(f"Recovery files retained at: {stage}", flush=True)
        if uploaded and stopped:
            done.touch()
    return 0 if uploaded and stopped and train_exit_code == 0 else 1


def completion_seen(log: Path, pattern: str) -> bool:
    needle = pattern.encode("utf-8")
    previous = b""
    with log.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            combined = previous + chunk
            if needle in combined:
                return True
            previous = combined[-len(needle):]
    return False


def stop_reason(log: Path, pattern: str, tag: str, now: float, started_at: float) -> str | None:
    start, done = markers(log, tag)
    if done.exists():
        return None
    if start.exists():
        return "epilogue did not finish" if now - start.stat().st_mtime > ORPHAN_SECONDS else None
    if not log.exists():
        return "training log never appeared" if now - started_at > STALE_SECONDS else None
    age = now - log.stat().st_mtime
    if completion_seen(log, pattern):
        return "epilogue never started" if age > ORPHAN_SECONDS else None
    return "training log is inactive" if age > STALE_SECONDS else None


def watchdog(log: Path, pattern: str, tag: str) -> int:
    container_credentials()
    if not pattern:
        raise ValueError("completion pattern cannot be empty")
    started_at = time.time()
    while True:
        time.sleep(CHECK_INTERVAL)
        _, done = markers(log, tag)
        if done.exists():
            return 0
        reason = stop_reason(log, pattern, tag, time.time(), started_at)
        if reason:
            print(f"WATCHDOG: {reason}; requesting stop, never destruction.", flush=True)
            if stop_self():
                return 0
            # Keep retrying at the normal interval if both stop transports fail.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    final = commands.add_parser("finish")
    final.add_argument("tag")
    final.add_argument("output", type=Path)
    final.add_argument("repo")
    final.add_argument("train_exit_code", type=int)
    final.add_argument("log", type=Path)
    final.add_argument("--keep-running", action="store_true", help="only after successful, verified smoke runs")
    final.add_argument("--dry-run", action="store_true")
    guard = commands.add_parser("watchdog")
    guard.add_argument("log", type=Path)
    guard.add_argument("pattern")
    guard.add_argument("tag")
    guard.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "finish":
            return finish(args.tag, args.output, args.repo, args.train_exit_code, args.log,
                          keep_running=args.keep_running, dry_run=args.dry_run)
        if args.dry_run:
            print("Would monitor logs and request instance stop after 60 minutes inactivity or 15 minutes unfinished finalization.")
            return 0
        return watchdog(args.log, args.pattern, args.tag)
    except (OSError, ValueError) as exc:
        print(f"Lifecycle error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
