"""Runtime configuration from environment variables (see README "Environment variables")."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    max_upload_mb: float = field(default_factory=lambda: _float("ROBOTRACE_MAX_UPLOAD_MB", 200))
    max_duration_s: float = field(default_factory=lambda: _float("ROBOTRACE_MAX_DURATION_S", 180))
    work_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("ROBOTRACE_WORK_DIR", Path(tempfile.gettempdir()) / "robotrace")))
    anthropic_model: str = field(default_factory=lambda: os.environ.get("ROBOTRACE_CLAUDE_MODEL", "claude-opus-5-5"))
    remote_url: str = field(default_factory=lambda: os.environ.get("ROBOTRACE_REMOTE_URL", "").rstrip("/"))
    remote_token: str = field(default_factory=lambda: os.environ.get("ROBOTRACE_REMOTE_TOKEN", ""))
    remote_timeout_s: float = field(default_factory=lambda: _float("ROBOTRACE_REMOTE_TIMEOUT_S", 300))
    # Frames sent per annotation request; bounded to control cost and latency.
    max_frames: int = field(default_factory=lambda: int(_float("ROBOTRACE_MAX_FRAMES", 24)))

    @property
    def has_anthropic_key(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def get_settings() -> Settings:
    return Settings()
