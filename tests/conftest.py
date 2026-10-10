import os
import sys
from pathlib import Path

# Keep tests hermetic: never load the developer's real .env (API keys) into the test process.
os.environ["ROBOTRACE_SKIP_DOTENV"] = "1"

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robotrace.schemas import Annotation, VideoMetadata  # noqa: E402
from robotrace.video_processor import SignalProfile  # noqa: E402


def ann(start, end, action="move the cube", id=None):
    kw = {"id": id} if id else {}
    return Annotation(start=start, end=end, action=action, **kw)


@pytest.fixture
def meta():
    return VideoMetadata(filename="t.mp4", sha256="0" * 64, duration_s=10.0, fps=10.0, frame_count=100,
                         width=64, height=48, size_bytes=1)


def make_signal(active_spans, duration=10.0, hz=5.0, subtle_spans=(), cut_at=()):
    """Synthetic SignalProfile: motion 5.0 inside active spans, 0.3 in subtle spans, 0.02 elsewhere."""
    step = 1 / hz
    times = np.arange(step, duration + 1e-9, step)
    motion = np.full_like(times, 0.02)
    for s, e in subtle_spans:
        motion[(times >= s) & (times <= e)] = 0.3
    for s, e in active_spans:
        motion[(times >= s) & (times <= e)] = 5.0
    cut = np.full_like(times, 0.01)
    for t in cut_at:
        cut[np.argmin(np.abs(times - t))] = 0.9
    return SignalProfile(times, motion, cut, step)


@pytest.fixture
def tiny_video(tmp_path):
    """3-second 64x48 video: static for 1s, a moving square 1-2s, static 2-3s."""
    p = tmp_path / "tiny.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(30):
        img = np.full((48, 64, 3), 200, np.uint8)
        x = 5 if i < 10 else (5 + (i - 10) * 4 if i < 20 else 45)
        cv2.rectangle(img, (x, 15), (x + 12, 27), (0, 0, 255), -1)
        w.write(img)
    w.release()
    return p
