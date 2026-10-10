"""Annotation adapters. Every adapter turns a video into timestamped `Annotation`s.

Adapters (select with `get_adapter(name)`):
  sample   precomputed SIMULATED annotations for the bundled synthetic sample only (SHA-256 checked)
  motion   deterministic motion-energy segmentation; CPU only; produces UNLABELED activity segments
  claude   frame-based vision-language annotation via the Claude API (needs ANTHROPIC_API_KEY)
  remote   HTTP call to a GPU notebook (Colab/Kaggle) running a video-LLM; see notebooks/ and README

Important limitation: the `claude` adapter sees a bounded set of still frames with timestamps, not
continuous video, so boundaries are quantised to the sampling interval and fast actions can be missed.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Optional, Protocol

from pydantic import BaseModel, Field, ValidationError

from .config import SAMPLE_DIR, Settings, get_settings
from .schemas import Annotation, AnnotationResult, EvidenceFrame, VideoMetadata
from .video_processor import SignalProfile, compute_signals, extract_frames, uniform_times


class AdapterUnavailable(RuntimeError):
    """The adapter cannot run in this environment (missing key, endpoint down, wrong video...)."""


class AnnotationAdapter(Protocol):
    name: str
    label: str

    def availability(self) -> tuple[bool, str]: ...

    def annotate(self, path: Path, meta: VideoMetadata, cache_dir: Path) -> AnnotationResult: ...


# --------------------------------------------------------------------------------------------- sample
class SampleAdapter:
    name = "sample"
    label = "Precomputed sample (simulated annotator)"

    def __init__(self, sample_dir: Path = SAMPLE_DIR):
        self.sample_dir = sample_dir

    def availability(self) -> tuple[bool, str]:
        ok = (self.sample_dir / "baseline_annotations.json").exists()
        return ok, "Only valid for the bundled synthetic sample video." if ok else "Sample files missing."

    def annotate(self, path: Path, meta: VideoMetadata, cache_dir: Path) -> AnnotationResult:
        data = json.loads((self.sample_dir / "baseline_annotations.json").read_text())
        if data.get("video_sha256") != meta.sha256:
            # Never return canned annotations for a video they were not made for.
            raise AdapterUnavailable("Precomputed annotations exist only for the bundled sample video.")
        anns = [Annotation(id=a["id"], start=a["start"], end=a["end"], action=a["action"],
                           source="simulated_annotator") for a in data["annotations"]]
        return AnnotationResult(
            annotations=anns, adapter=self.name, is_live_inference=False, is_precomputed_sample=True,
            adapter_detail="Simulated annotator output with deliberately injected errors (see data/sample).")


# --------------------------------------------------------------------------------------------- motion
class MotionSegmentationAdapter:
    """Segments the video into contiguous high-motion intervals. No semantics: labels are generic."""
    name = "motion"
    label = "Motion segmentation (CPU, unlabeled)"

    def __init__(self, min_len_s: float = 0.6, merge_gap_s: float = 0.4):
        self.min_len_s = min_len_s
        self.merge_gap_s = merge_gap_s

    def availability(self) -> tuple[bool, str]:
        return True, "Always available. Finds when motion happens, not what the action is."

    def segments(self, sig: SignalProfile) -> list[tuple[float, float, float]]:
        if sig.motion.size == 0:
            return []
        thr = sig.active_threshold
        active = sig.motion > thr
        segs, start = [], None
        for i, a in enumerate(active):
            t = float(sig.times[i])
            if a and start is None:
                start = t - sig.step
            if not a and start is not None:
                segs.append([max(start, 0.0), t - sig.step])
                start = None
        if start is not None:
            segs.append([max(start, 0.0), float(sig.times[-1])])
        merged: list[list[float]] = []
        for s in segs:
            if merged and s[0] - merged[-1][1] <= self.merge_gap_s:
                merged[-1][1] = s[1]
            else:
                merged.append(s)
        out = []
        for s, e in merged:
            if e - s >= self.min_len_s:
                out.append((round(s, 2), round(e, 2), sig.mean_motion(s, e) / max(thr, 1e-6)))
        return out

    def annotate(self, path: Path, meta: VideoMetadata, cache_dir: Path) -> AnnotationResult:
        sig = compute_signals(path, meta)
        anns = [Annotation(start=s, end=e, action=f"activity segment {i + 1} (unlabeled motion)",
                           confidence=None, source=self.name)
                for i, (s, e, _ratio) in enumerate(self.segments(sig))]
        return AnnotationResult(
            annotations=anns, adapter=self.name, is_live_inference=True,
            adapter_detail=f"OpenCV frame differencing at {1 / sig.step:.1f} Hz, adaptive threshold "
                           f"{sig.active_threshold:.2f} grey levels.",
            warnings=["Motion segmentation provides timing only; action descriptions are placeholders "
                      "for a reviewer or a vision-language model to fill in."])


# --------------------------------------------------------------------------------------------- claude
class _VLMSegment(BaseModel):
    start: float
    end: float
    action: str
    confidence: float = Field(ge=0, le=1)


class _VLMOutput(BaseModel):
    segments: list[_VLMSegment]


_SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {"segments": {"type": "array", "items": {
        "type": "object",
        "properties": {"start": {"type": "number"}, "end": {"type": "number"},
                       "action": {"type": "string"}, "confidence": {"type": "number"}},
        "required": ["start", "end", "action", "confidence"], "additionalProperties": False}}},
    "required": ["segments"], "additionalProperties": False,
}

_ANNOTATE_PROMPT = """You are annotating a robot manipulation video for a robotics training dataset.
You are given {n} still frames sampled from a {dur:.2f}-second video; each is preceded by its timestamp.
Identify the sequence of robot actions (e.g. "reach toward the red cube", "grasp the cup", "place the cup
in the bin"). For each action return start and end timestamps in seconds within [0, {dur:.2f}], a short
imperative description naming the object, and your confidence in [0,1].
Rules: segments must be in temporal order and must not overlap. Only describe what the frames show; do not
invent actions for periods where the robot is idle. Because you only see frames every ~{step:.2f}s,
place boundaries at the nearest frame timestamps."""


def _b64(path: str) -> str:
    return base64.standard_b64encode(Path(path).read_bytes()).decode()


def _frames_content(frames: list[EvidenceFrame]) -> list[dict]:
    content: list[dict] = []
    for f in frames:
        content.append({"type": "text", "text": f"t = {f.t:.2f}s"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                     "data": _b64(f.path)}})
    return content


def claude_json_call(settings: Settings, content: list[dict], schema: dict, max_tokens: int = 16000) -> dict:
    """One Claude request returning schema-constrained JSON. Raises AdapterUnavailable on failure."""
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise AdapterUnavailable("The 'anthropic' package is not installed.") from e
    client = anthropic.Anthropic(max_retries=2, timeout=180.0)
    kwargs = dict(model=settings.anthropic_model, max_tokens=max_tokens,
                  output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
                  messages=[{"role": "user", "content": content}],
                  betas=["server-side-fallback-2026-07-01"])
    try:
        try:
            resp = client.beta.messages.create(fallbacks="default", **kwargs)
        except TypeError:  # older SDK without the typed parameter
            resp = client.beta.messages.create(extra_body={"fallbacks": "default"}, **kwargs)
    except anthropic.AuthenticationError as e:
        raise AdapterUnavailable("Claude API rejected the credentials (check ANTHROPIC_API_KEY).") from e
    except anthropic.RateLimitError as e:
        raise AdapterUnavailable("Claude API rate limit reached; try again shortly.") from e
    except anthropic.APIStatusError as e:
        raise AdapterUnavailable(f"Claude API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise AdapterUnavailable("Could not reach the Claude API (network error).") from e
    if resp.stop_reason == "refusal":
        raise AdapterUnavailable("The model declined this request.")
    if resp.stop_reason == "max_tokens":
        raise AdapterUnavailable("Model output was truncated (max_tokens).")
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise AdapterUnavailable("Model returned malformed JSON.") from e


class ClaudeFrameAdapter:
    name = "claude"
    label = "Claude vision (frame-based, live API)"

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def availability(self) -> tuple[bool, str]:
        if not self.settings.has_anthropic_key:
            return False, "Set ANTHROPIC_API_KEY to enable live Claude annotation."
        return True, f"Model {self.settings.anthropic_model}; up to {self.settings.max_frames} frames per video."

    def annotate(self, path: Path, meta: VideoMetadata, cache_dir: Path) -> AnnotationResult:
        ok, why = self.availability()
        if not ok:
            raise AdapterUnavailable(why)
        n = int(min(self.settings.max_frames, max(4, round(meta.duration_s))))  # ~1 fps, bounded
        frames = extract_frames(path, meta, uniform_times(meta.duration_s, n), cache_dir / "frames")
        step = meta.duration_s / max(n, 1)
        content = [{"type": "text", "text": _ANNOTATE_PROMPT.format(n=len(frames), dur=meta.duration_s, step=step)},
                   *_frames_content(frames)]
        raw = claude_json_call(self.settings, content, _SEGMENT_SCHEMA)
        try:
            parsed = _VLMOutput.model_validate(raw)
        except ValidationError as e:
            raise AdapterUnavailable(f"Model output failed schema validation: {e.error_count()} errors.") from e
        anns = [Annotation(start=round(s.start, 2), end=round(s.end, 2), action=s.action.strip()[:200],
                           confidence=s.confidence, confidence_kind="model_self_reported", source=self.name)
                for s in parsed.segments]
        return AnnotationResult(
            annotations=anns, adapter=self.name, is_live_inference=True,
            adapter_detail=f"{self.settings.anthropic_model}, {len(frames)} frames at ~{step:.2f}s spacing.",
            warnings=[f"Frame-based approximation: boundaries are quantised to ~{step:.2f}s and actions "
                      "shorter than that may be missed. Confidence is self-reported, not calibrated."])


# --------------------------------------------------------------------------------------------- remote
class RemoteAdapter:
    """Calls the GPU notebook service. Contract (see docs/REMOTE_API.md):

    GET  {url}/health   -> {"status": "ok", "model": str}
    POST {url}/annotate multipart: file=<video>; header Authorization: Bearer <token>
         -> {"model": str, "segments": [{"start": s, "end": s, "action": str, "confidence": float|null}]}
    """
    name = "remote"
    label = "Remote GPU video-LLM (Colab/Kaggle)"

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.settings.remote_token}"} if self.settings.remote_token else {}

    def health(self) -> tuple[bool, str]:
        if not self.settings.remote_url:
            return False, "ROBOTRACE_REMOTE_URL is not set."
        import requests
        try:
            r = requests.get(f"{self.settings.remote_url}/health", headers=self._headers(), timeout=5)
            r.raise_for_status()
            body = r.json()
            return body.get("status") == "ok", f"Endpoint healthy, model: {body.get('model', '?')}"
        except Exception as e:  # noqa: BLE001 - any failure means unavailable
            return False, f"Endpoint unreachable ({type(e).__name__}). The notebook session may have stopped."

    def availability(self) -> tuple[bool, str]:
        return self.health()

    def annotate(self, path: Path, meta: VideoMetadata, cache_dir: Path) -> AnnotationResult:
        ok, why = self.health()
        if not ok:
            raise AdapterUnavailable(why)
        import requests
        try:
            with open(path, "rb") as f:
                r = requests.post(f"{self.settings.remote_url}/annotate", headers=self._headers(),
                                  files={"file": (meta.filename, f, "video/mp4")},
                                  timeout=self.settings.remote_timeout_s)
            r.raise_for_status()
            body = r.json()
        except requests.Timeout as e:
            raise AdapterUnavailable(f"Remote inference timed out after {self.settings.remote_timeout_s:.0f}s.") from e
        except (requests.RequestException, ValueError) as e:
            raise AdapterUnavailable(f"Remote inference failed: {type(e).__name__}.") from e
        try:
            segs = _VLMOutput.model_validate({"segments": [
                {**s, "confidence": s.get("confidence") if s.get("confidence") is not None else 0.0}
                for s in body.get("segments", [])]}).segments
        except (ValidationError, TypeError, AttributeError) as e:
            raise AdapterUnavailable("Remote service returned data that does not match the contract.") from e
        anns = [Annotation(start=s.start, end=s.end, action=s.action.strip()[:200], source=self.name)
                for s in segs]
        return AnnotationResult(annotations=anns, adapter=self.name, is_live_inference=True,
                                adapter_detail=f"Remote model: {body.get('model', 'unknown')}",
                                warnings=["Remote model output; confidence not reported."])


# --------------------------------------------------------------------------------------------- helpers
ADAPTERS = {"sample": SampleAdapter, "motion": MotionSegmentationAdapter,
            "claude": ClaudeFrameAdapter, "remote": RemoteAdapter}


def get_adapter(name: str) -> AnnotationAdapter:
    if name not in ADAPTERS:
        raise KeyError(f"Unknown adapter '{name}'")
    return ADAPTERS[name]()


def attach_evidence(annotations: list[Annotation], path: Path, meta: VideoMetadata, cache_dir: Path) -> int:
    """Attach start / middle / end frames to each annotation. Returns number of distinct frames sampled."""
    want: dict[str, list[float]] = {}
    for a in annotations:
        lo, hi = sorted((a.start, a.end))
        lo = min(max(lo, 0.0), meta.duration_s)
        hi = min(max(hi, 0.0), meta.duration_s)
        eps = 0.5 / meta.fps
        want[a.id] = [min(lo + eps, meta.duration_s), (lo + hi) / 2, max(hi - eps, 0.0)]
    all_times = sorted({t for ts in want.values() for t in ts})
    frames = extract_frames(path, meta, all_times, cache_dir / "frames")
    by_idx = {f.frame_index: f for f in frames}
    from .video_processor import time_to_frame
    for a in annotations:
        ev = []
        for t in want[a.id]:
            f = by_idx.get(time_to_frame(t, meta.fps, meta.frame_count))
            if f and all(e.frame_index != f.frame_index for e in ev):
                ev.append(f)
        a.evidence = ev
    return len(frames)


def run_annotation(adapter: AnnotationAdapter, path: Path, meta: VideoMetadata,
                   cache_dir: Path) -> tuple[AnnotationResult, float]:
    t0 = time.perf_counter()
    result = adapter.annotate(path, meta, cache_dir)
    return result, time.perf_counter() - t0


__all__ = ["AdapterUnavailable", "AnnotationAdapter", "ADAPTERS", "get_adapter", "attach_evidence",
           "run_annotation", "claude_json_call"]
