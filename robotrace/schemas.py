"""Typed data model shared by every RoboTrace component.

All timestamps are in SECONDS (float) relative to the start of the video.
Frame indices are 0-based indices into the decoded video stream.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "robotrace.v1"
TIME_UNIT = "seconds"


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class VideoMetadata(BaseModel):
    filename: str
    sha256: str
    duration_s: float = Field(ge=0)
    fps: float = Field(gt=0)
    frame_count: int = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    size_bytes: int = Field(ge=0)
    is_synthetic_sample: bool = False


class EvidenceFrame(BaseModel):
    """A sampled frame used as evidence. `path` is a server-local cache path (never user-supplied)."""
    t: float = Field(ge=0, description="timestamp in seconds")
    frame_index: int = Field(ge=0)
    path: Optional[str] = None


class Annotation(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("ann"))
    start: float = Field(description="start timestamp, seconds")
    end: float = Field(description="end timestamp, seconds")
    action: str
    evidence: list[EvidenceFrame] = Field(default_factory=list)
    # Model-reported score. NOT a calibrated probability; see `confidence_kind`.
    confidence: Optional[float] = None
    confidence_kind: Literal["none", "model_self_reported", "heuristic"] = "none"
    source: str = "unknown"  # adapter name, "human" for reviewer-added rows
    # Vision-model observations (empty for engines that do not produce them). These describe the model's
    # ORIGINAL claim and are not rewritten when a reviewer edits the annotation.
    objects: list[str] = Field(default_factory=list)
    visual_evidence: str = ""                       # what the model says it saw in the frames
    cited_frame_times: list[float] = Field(default_factory=list)  # sampled frames the model cited (validated)
    boundary_precision_s: Optional[float] = None    # start/end are approximate to +/- this; None = not stated
    # Who the model says performed the action: "robot", "human", "unclear" (ambiguous evidence), or ""
    # (not reported, e.g. non-vision engines). Never inferred beyond what the frames show; see
    # annotation_engine._canonical_actor. Added in prompt_version obs-v4; "" on anything from before.
    actor: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


class Severity(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


SEVERITY_WEIGHT = {Severity.high: 3, Severity.medium: 2, Severity.low: 1}


class FlagOrigin(str, Enum):
    rule = "deterministic_rule"      # pure timestamp / signal arithmetic
    signal = "visual_signal"         # deterministic pixel statistics (motion energy)
    model = "model_judgment"         # learned model output (VLM / CLIP)


class Flag(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("flag"))
    code: str                      # machine-readable, e.g. "TS_OVERLAP"
    category: str                  # "timestamp" | "evidence" | "boundary" | "missing_candidate"
    severity: Severity
    origin: FlagOrigin
    annotation_id: Optional[str] = None  # None for interval-level flags (e.g. missing-action candidate)
    start: Optional[float] = None
    end: Optional[float] = None
    reason: str                    # human-readable explanation
    details: dict = Field(default_factory=dict)  # raw numbers behind the reason
    resolved: bool = False
    resolution_note: str = ""

    @property
    def key(self) -> str:
        """Stable identity across QC re-runs, so reviewer resolutions survive re-checks."""
        span = "" if self.start is None else f"{self.start:.2f}-{self.end:.2f}"
        return f"{self.code}|{self.annotation_id or span}"


class ReviewStatus(str, Enum):
    unreviewed = "unreviewed"
    correct = "correct"
    incorrect = "incorrect"
    needs_review = "needs_review"


class ReviewEvent(BaseModel):
    at: str = Field(default_factory=utcnow)
    kind: str            # "status" | "edit" | "resolve_flag" | "add" | "delete" | "recheck"
    target_id: str
    payload: dict = Field(default_factory=dict)


class ProcessingInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    adapter: str
    adapter_detail: str = ""
    is_live_inference: bool
    is_precomputed_sample: bool = False
    started_at: str = Field(default_factory=utcnow)
    annotation_seconds: float = 0.0
    qc_seconds: float = 0.0
    sampled_frames: int = 0
    warnings: list[str] = Field(default_factory=list)
    model_summary: str = ""  # vision model's description of the whole scene, if the engine gives one
    provider: str = ""       # API provider that served the model, e.g. "OpenCode Go" (vision engines)
    model: str = ""          # model id requested from that provider
    prompt_version: str = ""  # version of the vision prompt used (see annotation_engine), if any
    sampling: dict = Field(default_factory=dict)  # frame-sampling config + plan actually used (vision engines)


class AnnotationResult(BaseModel):
    """What an annotation adapter returns."""
    model_config = ConfigDict(protected_namespaces=())
    annotations: list[Annotation]
    adapter: str
    adapter_detail: str = ""
    is_live_inference: bool
    is_precomputed_sample: bool = False
    warnings: list[str] = Field(default_factory=list)
    model_summary: str = ""
    provider: str = ""
    model: str = ""
    prompt_version: str = ""
    sampling: dict = Field(default_factory=dict)


class ReviewSession(BaseModel):
    """Complete state of one video's review. Baseline is never mutated."""
    model_config = ConfigDict(protected_namespaces=())
    schema_version: str = SCHEMA_VERSION
    time_unit: str = TIME_UNIT
    video: VideoMetadata
    processing: ProcessingInfo
    baseline: list[Annotation]          # adapter output, immutable
    annotations: list[Annotation]       # working copy edited by the reviewer
    initial_flags: list[Flag] = Field(default_factory=list)  # QC output before any review (frozen)
    flags: list[Flag] = Field(default_factory=list)          # current QC output (re-run after edits)
    signals: Optional[dict] = None                          # SignalProfile.to_dict(), for re-checks
    model_verdicts: dict[str, dict] = Field(default_factory=dict)  # annotation_id -> verifier output
    status: dict[str, ReviewStatus] = Field(default_factory=dict)
    notes: dict[str, str] = Field(default_factory=dict)
    events: list[ReviewEvent] = Field(default_factory=list)
    deleted_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fill_status(self):
        for a in self.annotations:
            self.status.setdefault(a.id, ReviewStatus.unreviewed)
        return self

    def get(self, ann_id: str) -> Annotation:
        for a in self.annotations:
            if a.id == ann_id:
                return a
        raise KeyError(ann_id)
