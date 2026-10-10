"""Temporary-file housekeeping for uploads and frame caches."""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Sized


def write_in_chunks(data: Sized, dest: Path, chunk_size: int = 1 << 20) -> int:
    """Write a bytes-like, sliceable buffer to `dest` in fixed-size chunks. Returns bytes written.

    Takes a buffer already held by the caller (e.g. Streamlit's UploadedFile.getbuffer()) rather than a
    stream, because that is what `st.file_uploader` provides; it avoids adding a SECOND full-size copy on
    top of what Streamlit already holds, one chunk at a time, instead of one `f.write(whole_buffer)` call.
    """
    written = 0
    with open(dest, "wb") as f:
        for i in range(0, len(data), chunk_size):
            chunk = bytes(data[i:i + chunk_size])
            f.write(chunk)
            written += len(chunk)
    return written


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
