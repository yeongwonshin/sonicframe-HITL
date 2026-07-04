from __future__ import annotations

import os
from pathlib import Path


class ConfigurationError(RuntimeError):
    """Raised when required production backends are not configured."""


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


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(f"Required environment variable {name} is not set")
    return value


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a float, got {raw!r}") from exc


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc


def csv_env(name: str, default: list[str] | None = None) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default or [])
    return [item.strip() for item in raw.split(",") if item.strip()]
