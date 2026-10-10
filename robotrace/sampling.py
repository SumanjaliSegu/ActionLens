"""Frame-sampling plans for the frame-based vision engines (`muse`, `gemini`) and deterministic merging of
per-window results.

Modes (select with ROBOTRACE_SAMPLING):
  legacy    (default) one request, ~1 frame/s uniformly over the whole video, at most max_frames frames.
            Spacing therefore grows with duration (24.6 s -> ~1.0 s, 40 s -> ~1.7 s, 180 s -> 7.5 s).
  windowed  fixed target spacing (ROBOTRACE_FRAME_SPACING_S) independent of duration: the video is split into
            overlapping windows of at most max_frames frames, one request per window, at most
            ROBOTRACE_MAX_WINDOWS requests (beyond that the spacing is widened, and a warning says so).
Frame timestamps are always absolute (seconds from the start of the video), so no window-relative
translation is needed; each window's actions are validated against that window's frames.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .schemas import Annotation
from .video_processor import uniform_times


@dataclass(frozen=True)
class SamplingWindow:
    start: float
    end: float
    times: list[float]


@dataclass(frozen=True)
class SamplingPlan:
    mode: str
    windows: list[SamplingWindow]
    spacing_s: float
    capped: bool = False
    config: dict = field(default_factory=dict)

    @property
    def n_requests(self) -> int:
        return len(self.windows)

    @property
    def n_frames(self) -> int:
        return sum(len(w.times) for w in self.windows)


def sampling_config(settings) -> dict:
    """The settings that determine which frames are sent. Part of a result's identity (caches, exports)."""
    if settings.sampling_mode != "windowed":
        return {"mode": "legacy", "max_frames": int(settings.max_frames)}
    return {"mode": "windowed", "max_frames": int(settings.max_frames), "spacing_s": float(settings.frame_spacing_s),
            "overlap_s": float(settings.window_overlap_s), "max_windows": int(settings.max_windows)}


def plan_sampling(duration: float, cfg: dict) -> SamplingPlan:
    mf = max(1, int(cfg["max_frames"]))
    if duration <= 0:
        return SamplingPlan(cfg["mode"], [], 0.0, config=cfg)
    if cfg["mode"] == "legacy":
        n = int(min(mf, max(4, round(duration))))
        return SamplingPlan("legacy", [SamplingWindow(0.0, duration, uniform_times(duration, n))], duration / n,
                            config=cfg)
    spacing = max(0.05, float(cfg["spacing_s"]))
    length = mf * spacing
    if duration <= length:
        n = int(min(mf, max(4, math.ceil(duration / spacing - 1e-9))))
        return SamplingPlan("windowed", [SamplingWindow(0.0, duration, uniform_times(duration, n))], duration / n,
                            config=cfg)
    overlap = min(max(0.0, float(cfg["overlap_s"])), length / 4)
    k = math.ceil((duration - overlap) / (length - overlap) - 1e-9)
    capped = k > int(cfg["max_windows"])
    if capped:
        k = max(1, int(cfg["max_windows"]))
        length = (duration + (k - 1) * overlap) / k
        spacing = length / mf
    windows = []
    for i in range(k):
        ws = round(i * (length - overlap), 3)
        we = duration if i == k - 1 else round(min(ws + length, duration), 3)
        n = int(min(mf, max(4, math.ceil((we - ws) / spacing - 1e-9))))
        windows.append(SamplingWindow(ws, we, [round(ws + t, 3) for t in uniform_times(we - ws, n)]))
    return SamplingPlan("windowed", windows, spacing, capped, config=cfg)


def describe_sampling(cfg: dict, duration: float | None = None) -> str:
    if duration:
        p = plan_sampling(duration, cfg)
        extra = " (spacing widened to stay within the request limit)" if p.capped else ""
        return (f"{cfg['mode']} sampling: {p.n_frames} frames at ~{p.spacing_s:.2f}s spacing, "
                f"{p.n_requests} model request(s){extra}")
    if cfg["mode"] == "legacy":
        return f"legacy sampling: ~1 frame/s, at most {cfg['max_frames']} frames, 1 model request"
    return (f"windowed sampling: ~{cfg['spacing_s']:.2f}s spacing, at most {cfg['max_frames']} frames per request, "
            f"at most {cfg['max_windows']} requests")


# ------------------------------------------------------------------------------------------- merging
_STOP = {"the", "a", "an", "to", "of", "into", "in", "on", "from", "with", "and", "it", "its", "toward", "towards",
         "over", "at", "onto", "inside"}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP}


def same_action(a: str, b: str, threshold: float = 0.6) -> bool:
    """Deterministic text match used only to de-duplicate the same action seen by two overlapping windows."""
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= threshold


def merge_window_annotations(per_window: list[list[Annotation]], tol: float) -> tuple[list[Annotation], int]:
    """Merge annotations from consecutive overlapping windows. Returns (annotations, number merged).

    Two annotations are merged only if they come from different windows, overlap in time (within tol) and
    describe the same action (same_action). Everything else is kept, including different actions that overlap
    at a seam, which QC then flags for review. Nothing is dropped because of which window it came from.
    """
    tagged = [(a, i) for i, anns in enumerate(per_window) for a in anns]
    tagged.sort(key=lambda x: (min(x[0].start, x[0].end), x[0].end, x[1]))
    out: list[tuple[Annotation, set[int]]] = []
    merged = 0
    for a, i in tagged:
        target = None
        if a.end > a.start:
            for k in range(len(out) - 1, -1, -1):
                p, srcs = out[k]
                if p.end + tol < a.start:
                    continue
                if i not in srcs and p.end > p.start and a.start <= p.end + tol and same_action(p.action, a.action):
                    target = k
                    break
        if target is None:
            out.append((a, {i}))
            continue
        p, srcs = out[target]
        p.start, p.end = min(p.start, a.start), max(p.end, a.end)
        p.objects = p.objects + [o for o in a.objects if o not in p.objects]
        p.cited_frame_times = sorted(set(p.cited_frame_times) | set(a.cited_frame_times))
        if a.visual_evidence and a.visual_evidence not in p.visual_evidence:
            p.visual_evidence = f"{p.visual_evidence} | {a.visual_evidence}".strip(" |")[:500]
        if p.confidence is not None and a.confidence is not None:
            p.confidence = min(p.confidence, a.confidence)  # conservative; never invented
        srcs.add(i)
        merged += 1
    result = [a for a, _ in out]
    result.sort(key=lambda a: (min(a.start, a.end), a.end))
    return result, merged
