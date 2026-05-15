"""Small logging helpers for alpha2rescore."""

from __future__ import annotations

import sys
from datetime import datetime


def log(message: str, component: str) -> None:
    """Emit one timestamped line to stderr and flush immediately."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{component}] {message}", file=sys.stderr, flush=True)


def format_duration(seconds: float) -> str:
    """Format a duration in seconds for progress logs."""
    return f"{seconds:.2f}s"
