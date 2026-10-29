"""Foreground execution and console/file logging for the local launchers."""
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


def resolve_executable(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise OSError(f"Executable not found: {name}")
    return resolved


def cancel_child(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    try:
        child.terminate()
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
    except ProcessLookupError:
        child.wait()


def run_foreground(command: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> int:
    child = subprocess.Popen(list(command), cwd=cwd, env=dict(env))
    try:
        return child.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        cancel_child(child)


def run_logged(command: Sequence[str], *, cwd: Path, env: Mapping[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Windows exclusive text open can follow a dangling symlink to a new target.
    if log_path.is_symlink():
        raise FileExistsError(f"Log path is an existing link: {log_path}; choose a fresh --log-file")
    # Open the log before starting a possibly paid job. A logging failure must
    # not leave an unobserved child process running.
    with log_path.open("x", encoding="utf-8") as log:
        child = subprocess.Popen(list(command), cwd=cwd, env=dict(env), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                 errors="replace", bufsize=1)
        try:
            assert child.stdout is not None
            for line in child.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            return child.wait()
        except KeyboardInterrupt:
            return 130
        finally:
            cancel_child(child)
            if child.stdout is not None:
                child.stdout.close()
