"""Helpers for invoking package modules under alternate Python environments."""

from __future__ import annotations

from collections import deque
import os
import subprocess
import sys
import threading
from pathlib import Path


MAX_CAPTURED_LOG_LINES = 200


def _forward_pipe(
    pipe,
    target_stream,
    prefix: str,
    line_buffer: deque[str],
) -> None:
    """Forward one child pipe to the parent stream while keeping a short tail."""
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            rendered = line.rstrip("\n")
            line_buffer.append(rendered)
            target_stream.write(f"{prefix}{rendered}\n")
            target_stream.flush()
    finally:
        pipe.close()


def run_python_module(
    python_executable: str,
    module: str,
    args: list[str],
    src_root: Path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a package module with PYTHONPATH pointed at the local source tree."""
    env = os.environ.copy()
    pythonpath_parts = [str(src_root)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    if extra_env:
        env.update(extra_env)

    command = [python_executable, "-m", module, *args]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )

    stdout_lines: deque[str] = deque(maxlen=MAX_CAPTURED_LOG_LINES)
    stderr_lines: deque[str] = deque(maxlen=MAX_CAPTURED_LOG_LINES)
    stdout_thread = threading.Thread(
        target=_forward_pipe,
        args=(process.stdout, sys.stdout, f"[{module} stdout] ", stdout_lines),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_forward_pipe,
        args=(process.stderr, sys.stderr, f"[{module} stderr] ", stderr_lines),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    return_code = process.wait()
    stdout_thread.join()
    stderr_thread.join()

    completed = subprocess.CompletedProcess(
        args=command,
        returncode=return_code,
        stdout="\n".join(stdout_lines),
        stderr="\n".join(stderr_lines),
    )

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail_parts = [f"{python_executable} -m {module} exited with code {completed.returncode}"]
        if stderr:
            detail_parts.append(f"stderr:\n{stderr}")
        if stdout:
            detail_parts.append(f"stdout:\n{stdout}")
        raise RuntimeError("\n\n".join(detail_parts))
    return completed
