"""Full-frame-rate motion analysis used to tighten the outer boundaries of vision-model annotations.

The vision model sees sparse keyframes; this module looks at EVERY decoded frame (the video's native rate, e.g.
15-30 fps) and finds when the robot first and last visibly moves. Rules = "variant B" of offline Experiment 2
(development-validated on 8 clips only): blurred frame difference > 8 grey levels, largest connected changed
region >= 0.05 % of the frame, motion sustained >= 0.2 s. Trimming only CUTS idle time at the very start of the
first segment and the very end of the last segment, keeping one frame of margin; it never extends a segment and
never touches interior boundaries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .schemas import Annotation

DIFF_THR = 8
AREA_FRAC = 0.0005
MIN_RUN_S = 0.2
MAX_WIDTH = 640


@dataclass
class MotionSpan:
    fps: float
    frames: int
    first_motion_s: Optional[float]
    last_motion_s: Optional[float]


def detect_motion_span(path: Path) -> MotionSpan:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    kernel = np.ones((3, 3), np.uint8)
    prev, moving = None, []
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if f.shape[1] > MAX_WIDTH:
                f = cv2.resize(f, (MAX_WIDTH, int(f.shape[0] * MAX_WIDTH / f.shape[1])), interpolation=cv2.INTER_AREA)
            g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            if prev is None:
                moving.append(False)
            else:
                m = cv2.morphologyEx((cv2.absdiff(g, prev) > DIFF_THR).astype(np.uint8), cv2.MORPH_OPEN, kernel)
                n, _, st, _ = cv2.connectedComponentsWithStats(m)
                moving.append(n > 1 and int(st[1:, cv2.CC_STAT_AREA].max()) >= AREA_FRAC * g.size)
            prev = g
    finally:
        cap.release()
    if fps <= 0 or not moving:
        return MotionSpan(fps, len(moving), None, None)
    k = max(1, math.ceil(MIN_RUN_S * fps - 1e-9))
    runs, s = [], None
    for i, m in enumerate(moving + [False]):
        if m and s is None:
            s = i
        elif not m and s is not None:
            if i - s >= k:
                runs.append((s, i - 1))
            s = None
    if not runs:
        return MotionSpan(fps, len(moving), None, None)
    return MotionSpan(fps, len(moving), runs[0][0] / fps, runs[-1][1] / fps)


def trim_idle(annotations: list[Annotation], span: MotionSpan, duration: float) -> dict:
    """Cut idle time before the first and after the last visible motion. Returns what was changed."""
    out = {"start_trimmed_s": 0.0, "end_trimmed_s": 0.0}
    valid = [a for a in annotations if a.end > a.start]
    if not valid or span.first_motion_s is None or span.fps <= 0:
        return out
    dt = 1.0 / span.fps
    cut_start = max(0.0, span.first_motion_s - dt)
    cut_end = min(duration, span.last_motion_s + dt)
    first = min(valid, key=lambda a: a.start)
    last = max(valid, key=lambda a: a.end)
    if cut_start > first.start and cut_start < first.end:
        out["start_trimmed_s"] = round(cut_start - first.start, 3)
        first.start = round(cut_start, 3)
    if cut_end < last.end and cut_end > last.start:
        out["end_trimmed_s"] = round(last.end - cut_end, 3)
        last.end = round(cut_end, 3)
    return out
