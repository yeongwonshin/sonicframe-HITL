from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    return Path(os.getenv("SONICFRAME_WORKSPACE", "workspace")).resolve()


def ensure_workspace() -> Path:
    root = workspace_root()
    for name in ["uploads", "exports", "projects"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def sample_fps() -> int:
    try:
        return max(1, int(os.getenv("SONICFRAME_SAMPLE_FPS", "6")))
    except ValueError:
        return 6


def default_style() -> str:
    return os.getenv("SONICFRAME_DEFAULT_STYLE", "balanced")
