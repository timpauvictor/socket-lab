"""Shared process helpers for Modal training jobs."""
from __future__ import annotations

import subprocess
import threading
from importlib.metadata import PackageNotFoundError, version


def package_versions() -> dict[str, str]:
    versions = {}
    for name in ("torch", "transformers", "peft", "datasets", "accelerate"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def run_streaming(command: list[str]) -> tuple[str, str, int]:
    """Forward stdout and stderr lines as they arrive and return both streams/status."""
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    assert proc.stdout is not None and proc.stderr is not None
    out_lines: list[str] = []
    err_lines: list[str] = []

    def forward(stream, lines: list[str]) -> None:
        for line in stream:
            lines.append(line)
            print(line, end="", flush=True)

    workers = (threading.Thread(target=forward, args=(proc.stdout, out_lines)),
               threading.Thread(target=forward, args=(proc.stderr, err_lines)))
    for worker in workers:
        worker.start()
    returncode = proc.wait()
    for worker in workers:
        worker.join()
    return "".join(out_lines), "".join(err_lines), returncode
