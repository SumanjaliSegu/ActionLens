"""Runtime configuration from environment variables (see README "Environment variables")."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> list[str]:
    """Load KEY=VALUE lines from a local .env into os.environ. Returns the names set (never values).

    Variables already present in the environment (shell, host secrets) always win. Blank lines, comments,
    an optional `export ` prefix and surrounding quotes are handled. Set ROBOTRACE_SKIP_DOTENV=1 to disable.
    """
    if os.environ.get("ROBOTRACE_SKIP_DOTENV") == "1" or not path.is_file():
        return []
    loaded = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.removeprefix("export ").split("=", 1)
        name, value = name.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name.isidentifier() and value and name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded


DOTENV_LOADED = load_dotenv()


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
    gemini_model: str = field(default_factory=lambda: os.environ.get("ROBOTRACE_GEMINI_MODEL", "gemini-3.8-flash"))
    gemini_timeout_s: float = field(default_factory=lambda: _float("ROBOTRACE_GEMINI_TIMEOUT_S", 180))
    opencode_go_url: str = field(default_factory=lambda: os.environ.get(
        "ROBOTRACE_OPENCODE_GO_URL", "https://opencode.ai/zen/go/v1").rstrip("/"))
    muse_model: str = field(default_factory=lambda: os.environ.get("ROBOTRACE_MUSE_MODEL",
                                                                   "muse-spark-1.3-contributor"))
    opencode_timeout_s: float = field(default_factory=lambda: _float("ROBOTRACE_OPENCODE_TIMEOUT_S", 180))
    # Frame sampling for the muse / gemini engines (see robotrace/sampling.py). "legacy" keeps the original
    # single-request ~1 frame/s plan; "windowed" uses a fixed spacing over overlapping windows.
    sampling_mode: str = field(default_factory=lambda: "windowed" if os.environ.get(
        "ROBOTRACE_SAMPLING", "legacy").strip().lower() == "windowed" else "legacy")
    frame_spacing_s: float = field(default_factory=lambda: _float("ROBOTRACE_FRAME_SPACING_S", 0.5))
    window_overlap_s: float = field(default_factory=lambda: _float("ROBOTRACE_WINDOW_OVERLAP_S", 1.0))
    max_windows: int = field(default_factory=lambda: int(_float("ROBOTRACE_MAX_WINDOWS", 4)))
    # Frames sent per annotation request; bounded to control cost and latency.
    max_frames: int = field(default_factory=lambda: int(_float("ROBOTRACE_MAX_FRAMES", 24)))

    @property
    def has_anthropic_key(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    @property
    def gemini_api_key(self) -> str:
        # Read on demand (not a dataclass field) so the key never appears in repr() or logs.
        return os.environ.get("GEMINI_API_KEY", "").strip()

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def opencode_go_api_key(self) -> str:
        return os.environ.get("OPENCODE_GO_API_KEY", "").strip()

    @property
    def has_opencode_go_key(self) -> bool:
        return bool(self.opencode_go_api_key)


def get_settings() -> Settings:
    return Settings()
