"""Shared process helpers for Modal training jobs."""
from __future__ import annotations

import subprocess
import sys
import threading


def pip_freeze() -> dict[str, str]:
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    frozen = {}
    for line in out.stdout.splitlines():
        if "==" in line:
            name, version = line.split("==", 1)
            frozen[name.strip()] = version.strip()
    return {key: frozen[key] for key in ("torch", "transformers", "peft", "datasets", "accelerate") if key in frozen}


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
