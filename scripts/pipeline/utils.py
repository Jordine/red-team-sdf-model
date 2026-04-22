"""Shared helpers for the training pipeline."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def timestamp() -> str:
    """Return a UTC timestamp suitable for run directories: '20260415_143022'."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def expand_path(raw: str, run_name: str | None = None) -> Path:
    """Expand ~ and {run_name} placeholders, return resolved Path."""
    s = raw.replace("{run_name}", run_name or "unnamed")
    return Path(s).expanduser().resolve()


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


class ConfigError(Exception):
    """Raised when a run config fails validation."""


class SSHError(Exception):
    """Raised on SSH/SCP failures."""


class TrainingError(Exception):
    """Raised when remote training fails."""
