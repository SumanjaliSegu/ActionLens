"""Reviewer actions. Every mutation goes through here, is validated, and is logged as a ReviewEvent."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .quality_engine import carry_resolutions, run_qc
from .schemas import Annotation, ReviewEvent, ReviewSession, ReviewStatus
from .video_processor import SignalProfile


class ReviewError(ValueError):
    pass


def signals_of(session: ReviewSession) -> Optional[SignalProfile]:
    s = session.signals
    if not s:
        return None
    return SignalProfile(np.asarray(s["times"], float), np.asarray(s["motion"], float),
                         np.asarray(s["cut"], float), float(s["step"]))


def recheck(session: ReviewSession, log: bool = True) -> None:
    """Re-run QC on the current annotations, preserving resolutions of flags that still apply."""
    new = run_qc(session.annotations, session.video.duration_s, signals_of(session), session.model_verdicts)
    session.flags = carry_resolutions(new, session.flags)
    if log:
        session.events.append(ReviewEvent(kind="recheck", target_id="*",
                                          payload={"open_flags": sum(not f.resolved for f in session.flags)}))


def set_status(session: ReviewSession, ann_id: str, status: ReviewStatus | str, note: str = "") -> None:
    session.get(ann_id)
    status = ReviewStatus(status)
    prev = session.status.get(ann_id, ReviewStatus.unreviewed)
    session.status[ann_id] = status
    if note:
        session.notes[ann_id] = note
    session.events.append(ReviewEvent(kind="status", target_id=ann_id,
                                      payload={"from": prev.value, "to": status.value, "note": note}))


_VALID_ACTORS = {"", "robot", "human", "unclear"}


def edit_annotation(session: ReviewSession, ann_id: str, *, start: Optional[float] = None,
                    end: Optional[float] = None, action: Optional[str] = None,
                    objects: Optional[list[str]] = None, actor: Optional[str] = None) -> Annotation:
    a = session.get(ann_id)
    new_objects = a.objects if objects is None else [o.strip()[:80] for o in objects if o.strip()]
    new_start = a.start if start is None else round(float(start), 3)
    new_end = a.end if end is None else round(float(end), 3)
    new_action = a.action if action is None else action.strip()
    new_actor = a.actor if actor is None else actor.strip().lower()
    if new_start < 0:
        raise ReviewError("Start must be ≥ 0.")
    if new_end <= new_start:
        raise ReviewError("End must be after start.")
    if new_end > session.video.duration_s + 0.05:
        raise ReviewError(f"End must be ≤ video length ({session.video.duration_s:.2f}s).")
    if not new_action:
        raise ReviewError("Action description cannot be empty.")
    if len(new_action) > 300:
        raise ReviewError("Action description is too long (max 300 characters).")
    if len(new_objects) > 10:
        raise ReviewError("At most 10 objects per annotation.")
    if new_actor not in _VALID_ACTORS:
        raise ReviewError(f"Actor must be one of: robot, human, unclear, or empty (got {new_actor!r}).")
    changes = {k: [old, new] for k, old, new in (("start", a.start, new_start), ("end", a.end, new_end),
                                                ("action", a.action, new_action),
                                                ("objects", a.objects, new_objects),
                                                ("actor", a.actor, new_actor)) if old != new}
    if not changes:
        return a
    a.start, a.end, a.action, a.objects, a.actor = new_start, new_end, new_action, new_objects, new_actor
    if a.source != "human":
        a.source = f"{a.source}+edited" if not a.source.endswith("+edited") else a.source
    session.events.append(ReviewEvent(kind="edit", target_id=ann_id, payload=changes))
    recheck(session, log=False)
    return a


def add_annotation(session: ReviewSession, start: float, end: float, action: str,
                   from_flag: Optional[str] = None) -> Annotation:
    if start < 0 or end <= start or end > session.video.duration_s + 0.05:
        raise ReviewError("Invalid interval for new annotation.")
    if not action.strip():
        raise ReviewError("Action description cannot be empty.")
    a = Annotation(start=round(start, 3), end=round(end, 3), action=action.strip(), source="human")
    session.annotations.append(a)
    session.annotations.sort(key=lambda x: (min(x.start, x.end), x.end))
    session.status[a.id] = ReviewStatus.correct
    session.events.append(ReviewEvent(kind="add", target_id=a.id,
                                      payload={"start": a.start, "end": a.end, "action": a.action,
                                               "from_flag": from_flag}))
    recheck(session, log=False)
    return a


def delete_annotation(session: ReviewSession, ann_id: str) -> None:
    a = session.get(ann_id)
    session.annotations = [x for x in session.annotations if x.id != ann_id]
    session.deleted_ids.append(ann_id)
    session.status[ann_id] = ReviewStatus.incorrect
    session.events.append(ReviewEvent(kind="delete", target_id=ann_id,
                                      payload={"start": a.start, "end": a.end, "action": a.action}))
    recheck(session, log=False)


def resolve_flag(session: ReviewSession, flag_id: str, note: str = "", resolved: bool = True) -> None:
    for f in session.flags:
        if f.id == flag_id:
            f.resolved = resolved
            f.resolution_note = note
            session.events.append(ReviewEvent(kind="resolve_flag", target_id=flag_id,
                                              payload={"code": f.code, "resolved": resolved, "note": note}))
            return
    raise ReviewError(f"Unknown flag {flag_id}")


def summary(session: ReviewSession) -> dict:
    """Metrics computed only from real session data."""
    status_counts = {s.value: 0 for s in ReviewStatus}
    for a in session.annotations:
        status_counts[session.status.get(a.id, ReviewStatus.unreviewed).value] += 1

    def flag_stats(flags):
        cats: dict[str, int] = {}
        for f in flags:
            cats[f.category] = cats.get(f.category, 0) + 1
        flagged = {f.annotation_id for f in flags if f.annotation_id and not f.resolved}
        return {"total": len(flags), "open": sum(not f.resolved for f in flags), "by_category": cats,
                "flagged_annotations": len(flagged)}

    kinds: dict[str, int] = {}
    for e in session.events:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    return {
        "annotations_baseline": len(session.baseline),
        "annotations_current": len(session.annotations),
        "status": status_counts,
        "reviewed": sum(v for k, v in status_counts.items() if k != "unreviewed"),
        "flags_initial": flag_stats(session.initial_flags),
        "flags_current": flag_stats(session.flags),
        "review_actions": kinds,
        "review_action_total": sum(v for k, v in kinds.items() if k != "recheck"),
        "edited": sum(1 for e in session.events if e.kind == "edit"),
        "added": kinds.get("add", 0),
        "deleted": len(session.deleted_ids),
        "annotation_seconds": session.processing.annotation_seconds,
        "qc_seconds": session.processing.qc_seconds,
    }
