"""Minimal .env loader shared by the agent, LLM clients, and tool servers."""

import os
from pathlib import Path


def load_env(path: Path = Path(__file__).with_name(".env")) -> None:
    """Load KEY=value lines (no extra dependency); existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), os.path.expandvars(value.strip()))
