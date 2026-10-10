"""Annotation adapters. Every adapter turns a video into timestamped `Annotation`s.

Adapters (select with `get_adapter(name)`):
  sample   precomputed SIMULATED annotations for the bundled synthetic sample only (SHA-256 checked)
  motion   deterministic motion-energy segmentation; CPU only; produces UNLABELED activity segments
  claude   frame-based vision-language annotation via the Claude API (needs ANTHROPIC_API_KEY)
  gemini   frame-based vision-language annotation via the Gemini API (needs GEMINI_API_KEY)
  muse     frame-based vision-language annotation via Muse Spark on OpenCode Go (needs OPENCODE_GO_API_KEY)
  remote   HTTP call to a GPU notebook (Colab/Kaggle) running a video-LLM; see notebooks/ and README

Important limitation: the `claude`, `gemini` and `muse` adapters see a bounded set of still frames with timestamps, not
continuous video, so boundaries are quantised to the sampling interval and fast actions can be missed.
"""
from __future__ import annotations

import base64
import json
import time
import uuid
from pathlib import Path
from typing import Optional, Protocol

from pydantic import BaseModel, Field, ValidationError

from .config import SAMPLE_DIR, Settings, get_settings
from .sampling import describe_sampling, merge_window_annotations, plan_sampling, sampling_config
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


# --------------------------------------------------------------------------------------------- gemini
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Version of the observation prompt below (shared by the `gemini` and `muse` engines). Recorded on every run as
# ProcessingInfo.prompt_version so results produced with different prompts are never mixed up.
#   obs-v1  original prompt
#   obs-v2  adds: split into distinct visible steps (never add unseen ones), exclude initial still frames,
#           hedge uncertain colours
#   obs-v3  obs-v2 + states the time range the frames cover; for multi-window sampling also says an action may
#           be in progress at the first / last frame
#   obs-v4  obs-v3 + embodiment-agnostic: a robot arm/gripper OR human hands can be the manipulator (egocentric
#           human demonstrations returned "no robot visible" and zero actions under obs-v3)
OBSERVATION_PROMPT_VERSION = "obs-v4"

_GEMINI_PROMPT = """You are annotating a manipulation video for a robotics training dataset. The manipulator may be
a robot arm / gripper OR human hands (e.g. a first-person human demonstration); annotate whichever is acting.
You are given {n} still frames sampled at ~{step:.2f}s spacing from {ws:.2f}s to {we:.2f}s of a {dur:.2f}-second
video. Each frame is preceded by its timestamp in seconds from the start of the video; those timestamps are the
only valid times.{partial}
1. In "scene_summary", describe in one or two sentences what is visible: the manipulator (robot arm / gripper or
   human hands), the objects, and the setting.
   Set "robot_visible" to true if a robot manipulator OR human hands manipulating objects are visible, and to false
   only if neither is visible in any frame.
2. In "actions", list the manipulation actions you can see, in temporal order and without overlap. If the
   manipulator is a human, say so in the action text (e.g. "right hand grasps the screwdriver").
   - When the frames show them, split the manipulation into its distinct steps, each as its own action:
     approach / reach toward the object; grasp (gripper or fingers close on it); lift / transport (object held
     and moving); place / release (object set down, gripper or hand opens); retract / return (empty gripper or
     hand moves away). Other actions (e.g. screw, insert, align, hold, use a tool) are equally valid.
     Merge steps only when the frames cannot separate them, and never add a step that no frame shows.
   - If the manipulator is still in the first frames, the first action starts at the first frame where it has
     visibly moved; do not include the still frames in it.
   - Name a colour only when it is clearly visible; if unsure, say so (e.g. "light green or yellow block").
   For each action give:
   - "first_frame_t" and "last_frame_t": timestamps of the first and last frame in which the action is visible,
     copied exactly from the frame labels (equal if the action is visible in only one frame);
   - "action": a short imperative description naming the object, e.g. "grasp the red cup";
   - "objects": the objects involved, as they appear, e.g. ["red cup", "grey bin"];
   - "visual_evidence": what you actually see in the frames that shows this action (gripper position, contact,
     object state change). Describe what is visible, not what you assume happened between frames;
   - "evidence_frame_times": timestamps of the frames that show that evidence;
   - "confidence": your confidence in [0, 1].
Only describe what the frames show. Do not invent actions for periods where the manipulator is idle or not
visible.
If you cannot identify any action, return an empty "actions" list."""

_GEMINI_ACTION_FIELDS = ["first_frame_t", "last_frame_t", "action", "objects", "visual_evidence",
                         "evidence_frame_times", "confidence"]
# Gemini responseSchema (OpenAPI subset).
_GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "scene_summary": {"type": "STRING"},
        "robot_visible": {"type": "BOOLEAN"},
        "actions": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {"first_frame_t": {"type": "NUMBER"}, "last_frame_t": {"type": "NUMBER"},
                           "action": {"type": "STRING"},
                           "objects": {"type": "ARRAY", "items": {"type": "STRING"}},
                           "visual_evidence": {"type": "STRING"},
                           "evidence_frame_times": {"type": "ARRAY", "items": {"type": "NUMBER"}},
                           "confidence": {"type": "NUMBER"}},
            "required": _GEMINI_ACTION_FIELDS, "propertyOrdering": _GEMINI_ACTION_FIELDS}},
    },
    "required": ["scene_summary", "robot_visible", "actions"],
    "propertyOrdering": ["scene_summary", "robot_visible", "actions"],
}


class _GeminiAction(BaseModel):
    first_frame_t: float
    last_frame_t: float
    action: str = Field(min_length=1)
    objects: list[str] = Field(default_factory=list)
    visual_evidence: str = ""
    evidence_frame_times: list[float] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class _GeminiOutput(BaseModel):
    scene_summary: str = ""
    robot_visible: bool
    actions: list[_GeminiAction]


def _gemini_http_error(status: int, body_text: str, model: str, key: str) -> str:
    """Readable error for a non-200 response. The API key is redacted from anything echoed back."""
    try:
        err = json.loads(body_text).get("error", {})
        msg, code = str(err.get("message", "")), str(err.get("status", ""))
    except (ValueError, AttributeError):
        msg, code = "", ""
    if key:
        msg = msg.replace(key, "***")
    if status == 400 and ("API_KEY_INVALID" in body_text or "API key" in msg):
        hint = "Gemini rejected the API key (check GEMINI_API_KEY)."
    elif status in (401, 403):
        hint = "Gemini denied access (invalid key, API not enabled for the key's project, or unsupported region)."
    elif status == 404:
        hint = f"Gemini model '{model}' was not found; set ROBOTRACE_GEMINI_MODEL to a model your key can use."
    elif status == 429:
        hint = "Gemini quota or rate limit reached; wait and retry, or check the key's plan."
    elif status >= 500:
        hint = "Gemini service error; retry shortly."
    else:
        hint = "Gemini API error."
    return f"{hint} [HTTP {status}{' ' + code if code else ''}] {msg[:300]}".strip()


def parse_gemini_response(resp: dict) -> tuple[dict, dict]:
    """Extract the JSON answer and response metadata from a generateContent body. Raises AdapterUnavailable."""
    block = (resp.get("promptFeedback") or {}).get("blockReason")
    if block:
        raise AdapterUnavailable(f"Gemini blocked the request (blockReason={block}).")
    cands = resp.get("candidates") or []
    if not cands:
        raise AdapterUnavailable("Gemini returned no candidates.")
    reason = cands[0].get("finishReason") or "STOP"
    if reason == "MAX_TOKENS":
        raise AdapterUnavailable("Gemini output was truncated (MAX_TOKENS).")
    if reason != "STOP":
        raise AdapterUnavailable(f"Gemini stopped without a usable answer (finishReason={reason}).")
    parts = (cands[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    if text.startswith("```"):  # tolerate a markdown fence around otherwise valid JSON
        text = text.strip("`").removeprefix("json").strip()
    if not text:
        raise AdapterUnavailable("Gemini returned an empty answer.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise AdapterUnavailable("Gemini returned malformed JSON.") from e
    if not isinstance(data, dict):
        raise AdapterUnavailable("Gemini returned JSON that is not an object.")
    usage = resp.get("usageMetadata") or {}
    meta = {"model_version": resp.get("modelVersion", ""), "response_id": resp.get("responseId", ""),
            "prompt_tokens": usage.get("promptTokenCount"), "output_tokens": usage.get("candidatesTokenCount")}
    return data, meta


def gemini_json_call(settings: Settings, parts: list[dict], schema: dict,
                     max_output_tokens: int = 16384) -> tuple[dict, dict]:
    """One generateContent request returning (JSON answer, metadata). Raises AdapterUnavailable on failure."""
    key = settings.gemini_api_key
    if not key:
        raise AdapterUnavailable("Set GEMINI_API_KEY to enable live Gemini annotation.")
    import requests
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema,
                                 "maxOutputTokens": max_output_tokens}}
    try:
        r = requests.post(GEMINI_ENDPOINT.format(model=settings.gemini_model), json=body,
                          headers={"x-goog-api-key": key}, timeout=settings.gemini_timeout_s)
    except requests.Timeout as e:
        raise AdapterUnavailable(f"Gemini API timed out after {settings.gemini_timeout_s:.0f}s.") from e
    except requests.RequestException as e:
        raise AdapterUnavailable(f"Could not reach the Gemini API ({type(e).__name__}).") from e
    if r.status_code != 200:
        raise AdapterUnavailable(_gemini_http_error(r.status_code, r.text, settings.gemini_model, key))
    try:
        resp = r.json()
    except ValueError as e:
        raise AdapterUnavailable("Gemini API returned a non-JSON response.") from e
    return parse_gemini_response(resp)


def gemini_actions_to_annotations(out: _GeminiOutput, frame_times: list[float], duration: float,
                                  step: float, source: str = "gemini") -> tuple[list[Annotation], int]:
    """Convert frame-level observations to interval annotations. Returns (annotations, dropped citations).

    Each sampled frame stands for +/- step/2 around its timestamp, so an action first seen at frame t0 and
    last seen at t1 becomes [t0 - step/2, t1 + step/2] with boundary_precision_s = step/2. Reversed
    intervals are kept exactly as reported so QC flags them. Cited frame times that do not match a frame
    that was actually sent are dropped (the model invented them) and counted.
    """
    half = step / 2
    tol = max(step / 4, 0.05)
    dropped = 0
    anns = []
    for act in out.actions:
        t0, t1 = act.first_frame_t, act.last_frame_t
        start, end = (max(t0 - half, 0.0), min(t1 + half, duration)) if t1 >= t0 else (t0, t1)
        cited = set()
        for t in act.evidence_frame_times:
            near = min(frame_times, key=lambda f: abs(f - t), default=None)
            if near is not None and abs(near - t) <= tol:
                cited.add(near)
            else:
                dropped += 1
        anns.append(Annotation(
            start=round(start, 2), end=round(end, 2), action=act.action.strip()[:200],
            confidence=act.confidence, confidence_kind="model_self_reported", source=source,
            objects=[o.strip()[:80] for o in act.objects if o.strip()][:10],
            visual_evidence=act.visual_evidence.strip()[:500], cited_frame_times=sorted(cited),
            boundary_precision_s=round(half, 2)))
    return anns, dropped


_PARTIAL_WINDOW_NOTE = ("\nThese frames cover only part of the video: an action may already be in progress at the "
                        "first frame or continue after the last frame; report only what these frames show.")


def _ground_actions(out: "_GeminiOutput", frame_times: list[float], step: float,
                    window: Optional[tuple[float, float]] = None) -> tuple[list, int]:
    """Keep only actions whose first/last timestamps are frames that were actually sent. Returns (kept, rejected).

    A time between the window's edge and its first / last sent frame (e.g. "0.0" when the first frame is at
    0.5 s) is the edge convention: it is accepted, clamped to the window edge, which gives exactly the interval
    produced before this check existed. Any other time that matches no sent frame is rejected (invented).
    """
    tol = max(step / 4, 0.05)
    lo, hi = min(frame_times), max(frame_times)
    ws, we = window if window else (lo, hi)

    def ground(t: float) -> Optional[float]:
        if ws - tol <= t < lo:
            return max(t, ws)
        if hi < t <= we + tol:
            return min(t, we)
        return t if any(abs(t - f) <= tol for f in frame_times) else None

    kept = []
    for act in out.actions:
        t0, t1 = ground(act.first_frame_t), ground(act.last_frame_t)
        if t0 is not None and t1 is not None:
            kept.append(act.model_copy(update={"first_frame_t": t0, "last_frame_t": t1}))
    return kept, len(out.actions) - len(kept)


def _annotate_frame_windows(settings: Settings, path: Path, meta: VideoMetadata, cache_dir: Path, *,
                            adapter: str, provider: str, model: str, call,
                            extra_warnings: Optional[list[str]] = None) -> AnnotationResult:
    """Shared runner for frame-based vision engines: plan -> per-window request -> validate -> merge.

    `call(prompt, frames)` performs ONE request and returns (json_dict, info). Any failure in any window raises
    AdapterUnavailable for the whole run (no partial results, no automatic retries).
    """
    cfg = sampling_config(settings)
    plan = plan_sampling(meta.duration_s, cfg)
    if not plan.windows:
        raise AdapterUnavailable("No frames could be decoded from this video.")
    multi = plan.n_requests > 1
    per_window, infos, summaries, visible, object_sets, windows_used = [], [], [], [], [], []
    window_details: list[dict] = []
    n_frames = dropped = rejected = 0
    step = plan.spacing_s
    # 1) decode every window's frames locally; 2) send the model requests (in parallel when configured);
    # 3) validate and convert in window order. Any failure fails the whole run (no partial results, no retries).
    prepared = []
    for w in plan.windows:
        frames = extract_frames(path, meta, w.times, cache_dir / "frames")
        if not frames:
            raise AdapterUnavailable("No frames could be decoded from this video.")
        # legacy keeps its original step definition; windowed uses the frames actually decoded in the window
        wstep = meta.duration_s / len(w.times) if plan.mode == "legacy" else (w.end - w.start) / len(frames)
        prompt = _GEMINI_PROMPT.format(n=len(frames), dur=meta.duration_s, step=wstep, ws=w.start, we=w.end,
                                       partial=_PARTIAL_WINDOW_NOTE if multi else "")
        prepared.append((w, frames, wstep, prompt))
    workers = max(1, min(int(getattr(settings, "parallel_requests", 1)), len(prepared)))
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(call, prompt, frames) for _, frames, _, prompt in prepared]
            responses = []
            for fut in futures:  # in window order; the first failure (by window) is reported
                responses.append(fut.result())
    else:
        responses = [call(prompt, frames) for _, frames, _, prompt in prepared]
    for (w, frames, step, _), (data, info) in zip(prepared, responses):
        try:
            parsed = _GeminiOutput.model_validate(data)
        except ValidationError as e:
            raise AdapterUnavailable(f"{provider} output failed schema validation: {e.error_count()} errors.") from e
        ftimes = [f.t for f in frames]
        kept, rej = _ground_actions(parsed, ftimes, step, window=(w.start, w.end))
        if parsed.actions and not kept:
            raise AdapterUnavailable(f"{provider} returned {len(parsed.actions)} action(s) but none used the "
                                     "timestamps of frames that were sent; output rejected.")
        anns, d = gemini_actions_to_annotations(parsed.model_copy(update={"actions": kept}), ftimes,
                                                meta.duration_s, step, source=adapter)
        per_window.append(anns)
        infos.append(info)
        summaries.append(parsed.scene_summary.strip())
        visible.append(parsed.robot_visible)
        object_sets.append(sorted({o.lower() for a_ in anns for o in a_.objects}))
        windows_used.append([w.start, w.end, len(frames)])
        window_details.append({"window_s": [w.start, w.end], "frames": len(frames),
                               "manipulator_visible": parsed.robot_visible, "actions_returned": len(parsed.actions),
                               "actions_rejected": rej, "scene_summary": parsed.scene_summary.strip()[:300],
                               "response_id": info.get("response_id")})
        n_frames += len(frames)
        dropped += d
        rejected += rej
    anns, n_merged = merge_window_annotations(per_window, tol=step / 2) if multi else (per_window[0], 0)

    shown_step = step if not multi else plan.spacing_s
    warnings = [f"Frame-based approximation: the model saw {n_frames} still frames ~{shown_step:.2f}s apart, so "
                f"segment boundaries are approximate (±{shown_step / 2:.2f}s from sampling alone) and actions "
                "shorter than the spacing may be missed. Confidence is self-reported, not calibrated."]
    if multi:
        warnings.append(f"Analysed in {plan.n_requests} overlapping windows ({plan.n_requests} model requests); "
                        f"{n_merged} duplicate segment(s) at window seams were merged.")
    if plan.capped:
        warnings.append(f"The video is too long for {cfg['spacing_s']:.2f}s spacing within {cfg['max_windows']} "
                        f"requests; spacing was widened to ~{plan.spacing_s:.2f}s.")
    if not any(visible):
        warnings.append(f"{provider} ({model}) reported that no robot or manipulating hands are visible in the "
                        "sampled frames.")
    elif not all(visible):
        warnings.append(f"Windows disagree on whether a manipulator is visible ({sum(visible)}/{len(visible)} say yes).")
    if multi and len({tuple(s) for s in object_sets if s}) > 1:
        warnings.append("Object names differ between windows: " + "; ".join(", ".join(s) for s in object_sets if s)
                        + ". Check that they refer to the same objects.")
    if rejected:
        warnings.append(f"{rejected} action(s) used timestamps that match no frame that was sent and were rejected.")
    if dropped:
        warnings.append(f"{dropped} cited frame timestamp(s) did not match any frame that was sent and "
                        "were discarded.")
    warnings += extra_warnings or []
    ids = [i.get("response_id") or "unreported" for i in infos]
    tin = [i.get("prompt_tokens") for i in infos]
    tout = [i.get("output_tokens") for i in infos]
    tokens = (f", tokens in/out {sum(tin)}/{sum(x or 0 for x in tout)}" if all(x is not None for x in tin) else "")
    served = "/".join(sorted({i.get("model_version") or "unreported" for i in infos}))
    frames_txt = (f"{n_frames} frames at ~{step:.2f}s spacing" if not multi else
                  f"{n_frames} frames in {plan.n_requests} windows at ~{plan.spacing_s:.2f}s spacing")
    detail = f"{provider} · {model} (served as {served}), {frames_txt}, response id {'+'.join(ids)}{tokens}."
    summary = " | ".join(dict.fromkeys(x for x in summaries if x))[:600]
    sampling = {"config": cfg, "mode": plan.mode, "spacing_s": round(plan.spacing_s, 3), "requests": plan.n_requests,
                "frames": n_frames, "windows": windows_used, "capped": plan.capped, "merged_at_seams": n_merged,
                "rejected_actions": rejected, "window_details": window_details}
    return AnnotationResult(annotations=anns, adapter=adapter, is_live_inference=True, adapter_detail=detail,
                            warnings=warnings, model_summary=summary, provider=provider, model=model,
                            prompt_version=OBSERVATION_PROMPT_VERSION, sampling=sampling)


class GeminiFrameAdapter:
    name = "gemini"
    label = "Gemini vision (frame-based, live API)"

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def availability(self) -> tuple[bool, str]:
        if not self.settings.has_gemini_key:
            return False, "Set GEMINI_API_KEY to enable live Gemini annotation."
        return True, f"Google Gemini · {self.settings.gemini_model}; {describe_sampling(sampling_config(self.settings))}."

    def annotate(self, path: Path, meta: VideoMetadata, cache_dir: Path) -> AnnotationResult:
        ok, why = self.availability()
        if not ok:
            raise AdapterUnavailable(why)

        def call(prompt: str, frames: list[EvidenceFrame]):
            parts: list[dict] = [{"text": prompt}]
            for f in frames:
                parts.append({"text": f"t = {f.t:.2f}s"})
                parts.append({"inlineData": {"mimeType": "image/jpeg", "data": _b64(f.path)}})
            return gemini_json_call(self.settings, parts, _GEMINI_SCHEMA)

        return _annotate_frame_windows(self.settings, path, meta, cache_dir, adapter=self.name,
                                       provider="Google Gemini", model=self.settings.gemini_model, call=call)


# ------------------------------------------------------------------------------ muse spark (opencode go)
# OpenCode Go serves Muse Spark through an OpenAI Responses-style endpoint: POST {base}/responses with
# "Authorization: Bearer <key>". Go asks clients to send their own User-Agent and a stable x-opencode-session.
_MUSE_JSON_ONLY = """
Respond with ONLY one JSON object (no prose, no markdown fences) of exactly this shape:
{"scene_summary": string, "robot_visible": boolean, "actions": [{"first_frame_t": number,
 "last_frame_t": number, "action": string, "objects": [string], "visual_evidence": string,
 "evidence_frame_times": [number], "confidence": number}]}"""

MUSE_DATA_NOTICE = ("Data use: OpenCode's docs state that Muse Spark contributor models may use prompts and "
                    "completions (here, the sampled video frames) to train future Meta models, and that they are "
                    "not zero-data-retention.")

_OPENCODE_SESSION = uuid.uuid4().hex  # stable for the life of this process


def _json_object(text: str, provider: str) -> dict:
    """Parse a model's JSON answer. Tolerates a markdown fence or surrounding prose; never invents content."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    if not text:
        raise AdapterUnavailable(f"{provider} returned an empty answer.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("{"), text.rfind("}")
        try:
            data = json.loads(text[i:j + 1]) if 0 <= i < j else None
        except json.JSONDecodeError:
            data = None
        if data is None:
            raise AdapterUnavailable(f"{provider} returned malformed JSON.") from None
    if not isinstance(data, dict):
        raise AdapterUnavailable(f"{provider} returned JSON that is not an object.")
    return data


def _opencode_http_error(status: int, body_text: str, model: str, key: str) -> str:
    """Readable error for a non-200 OpenCode Go response. The API key is redacted from anything echoed back."""
    try:
        err = json.loads(body_text).get("error") or {}
        msg = str(err.get("message", "")) if isinstance(err, dict) else str(err)
        etype = str(err.get("type", "")) if isinstance(err, dict) else ""
    except (ValueError, AttributeError):
        msg, etype = "", ""
    if key:
        msg = msg.replace(key, "***")
    if status == 401:
        hint = "OpenCode Go rejected the API key (check OPENCODE_GO_API_KEY)."
    elif status in (402, 403):
        hint = ("OpenCode Go refused the request (inactive Go subscription, usage limit reached, or a region "
                "not permitted for Muse Spark).")
    elif status == 404:
        hint = f"OpenCode Go model '{model}' was not found; set ROBOTRACE_MUSE_MODEL to a model listed for Go."
    elif status == 429:
        hint = "OpenCode Go usage or rate limit reached; wait and retry, or check your Go plan."
    elif status == 400:
        hint = "OpenCode Go rejected the request (the model may not accept this input format)."
    elif status >= 500:
        hint = "OpenCode Go service error; retry shortly."
    else:
        hint = "OpenCode Go API error."
    return f"{hint} [HTTP {status}{' ' + etype if etype else ''}] {msg[:300]}".strip()


def parse_opencode_response(resp: dict) -> tuple[dict, dict]:
    """Extract the JSON answer and metadata from a Responses-API body. Raises AdapterUnavailable."""
    err = resp.get("error")
    if err:
        msg = err.get("message", "") if isinstance(err, dict) else str(err)
        raise AdapterUnavailable(f"OpenCode Go returned an error: {str(msg)[:300]}")
    status = resp.get("status") or "completed"
    if status == "incomplete":
        reason = (resp.get("incomplete_details") or {}).get("reason") or "unknown"
        if reason == "max_output_tokens":
            raise AdapterUnavailable("Muse Spark output was truncated (max_output_tokens).")
        raise AdapterUnavailable(f"Muse Spark returned an incomplete answer (reason={reason}).")
    if status != "completed":
        raise AdapterUnavailable(f"Muse Spark response did not complete (status={status}).")
    texts, refusals = [], []
    for item in resp.get("output") or []:
        if item.get("type") != "message":
            continue  # reasoning items etc.
        for c in item.get("content") or []:
            if c.get("type") == "output_text":
                texts.append(c.get("text", ""))
            elif c.get("type") == "refusal":
                refusals.append(c.get("refusal", ""))
    text = "".join(texts)
    if refusals and not text.strip():
        raise AdapterUnavailable(f"Muse Spark declined the request: {' '.join(refusals)[:300]}")
    data = _json_object(text, "Muse Spark")
    usage = resp.get("usage") or {}
    meta = {"model_version": resp.get("model", ""), "response_id": resp.get("id", ""),
            "prompt_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}
    return data, meta


def opencode_go_call(settings: Settings, content: list[dict], max_output_tokens: int = 16000) -> tuple[dict, dict]:
    """One OpenCode Go /responses request returning (JSON answer, metadata). Raises AdapterUnavailable."""
    key = settings.opencode_go_api_key
    if not key:
        raise AdapterUnavailable("Set OPENCODE_GO_API_KEY to enable Muse Spark via OpenCode Go.")
    import requests

    from . import __version__
    body = {"model": settings.muse_model, "input": [{"role": "user", "content": content}],
            "max_output_tokens": max_output_tokens}
    headers = {"Authorization": f"Bearer {key}", "User-Agent": f"RoboTrace/{__version__}",
               "x-opencode-session": _OPENCODE_SESSION}
    try:
        r = requests.post(f"{settings.opencode_go_url}/responses", json=body, headers=headers,
                          timeout=settings.opencode_timeout_s)
    except requests.Timeout as e:
        raise AdapterUnavailable(f"OpenCode Go timed out after {settings.opencode_timeout_s:.0f}s.") from e
    except requests.RequestException as e:
        raise AdapterUnavailable(f"Could not reach OpenCode Go ({type(e).__name__}).") from e
    if r.status_code != 200:
        raise AdapterUnavailable(_opencode_http_error(r.status_code, r.text, settings.muse_model, key))
    try:
        resp = r.json()
    except ValueError as e:
        raise AdapterUnavailable("OpenCode Go returned a non-JSON response.") from e
    return parse_opencode_response(resp)


class MuseSparkAdapter:
    name = "muse"
    label = "Muse Spark via OpenCode Go (frame-based, live API)"

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def availability(self) -> tuple[bool, str]:
        if not self.settings.has_opencode_go_key:
            return False, "Set OPENCODE_GO_API_KEY to enable Muse Spark via OpenCode Go."
        return True, f"OpenCode Go · {self.settings.muse_model}; {describe_sampling(sampling_config(self.settings))}."

    def annotate(self, path: Path, meta: VideoMetadata, cache_dir: Path) -> AnnotationResult:
        ok, why = self.availability()
        if not ok:
            raise AdapterUnavailable(why)

        def call(prompt: str, frames: list[EvidenceFrame]):
            content: list[dict] = [{"type": "input_text", "text": prompt + _MUSE_JSON_ONLY}]
            for f in frames:
                content.append({"type": "input_text", "text": f"t = {f.t:.2f}s"})
                content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{_b64(f.path)}"})
            return opencode_go_call(self.settings, content)

        extra = [MUSE_DATA_NOTICE] if "contributor" in self.settings.muse_model else []
        return _annotate_frame_windows(self.settings, path, meta, cache_dir, adapter=self.name,
                                       provider="OpenCode Go", model=self.settings.muse_model, call=call,
                                       extra_warnings=extra)


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
            "claude": ClaudeFrameAdapter, "gemini": GeminiFrameAdapter, "muse": MuseSparkAdapter,
            "remote": RemoteAdapter}


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
           "run_annotation", "claude_json_call", "gemini_json_call", "parse_gemini_response",
           "gemini_actions_to_annotations", "OBSERVATION_PROMPT_VERSION", "opencode_go_call", "parse_opencode_response",
           "MUSE_DATA_NOTICE"]
