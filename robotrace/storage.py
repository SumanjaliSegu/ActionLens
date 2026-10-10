"""Temporary-file housekeeping for uploads and frame caches."""
from __future__ import annotations

import shutil
import time
from pathlib import Path


def cleanup_old_files(directory: Path, max_age_hours: float = 6.0) -> int:
    """Delete files/subdirectories in `directory` older than `max_age_hours`. Returns count removed."""
    if not directory.exists():
        return 0
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for p in directory.iterdir():
        try:
            if p.stat().st_mtime < cutoff:
                shutil.rmtree(p) if p.is_dir() else p.unlink()
                removed += 1
        except OSError:
            continue
    return removed
