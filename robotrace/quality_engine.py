"""Annotation quality control.

Three clearly separated layers, each tagged on every flag via `Flag.origin`:
  1. deterministic_rule  - pure timestamp arithmetic (no pixels, no models)
  2. visual_signal       - deterministic pixel statistics (motion energy, histogram cut score)
  3. model_judgment      - optional learned-model verification (Claude vision verifier)

Nothing here edits annotations. Gaps and uncovered motion are reported as *candidates* only.
Review priority is a transparent weighted count of unresolved flags, not a probability.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .schemas import SEVERITY_WEIGHT, Annotation, Flag, FlagOrigin, Severity
from .video_processor import SignalProfile


@dataclass(frozen=True)
class QCConfig:
    tolerance_s: float = 0.05          # float slack for overlaps / range checks
    gap_s: float = 1.0                 # report uncovered intervals longer than this
    min_len_s: float = 0.25            # shorter segments are suspicious
    long_factor: float = 2.0           # longer than factor x median duration ...
    long_min_s: float = 4.0            # ... and longer than this many seconds
    low_activity_frac: float = 0.2     # < this fraction of active steps => weak visual support
    idle_run_s: float = 1.0            # an idle stretch this long inside a segment is suspicious
    boundary_window_s: float = 0.6     # look-around window for boundary checks
    missing_active_frac: float = 0.5   # uncovered interval this active => missing-action candidate
    missing_min_s: float = 0.5
    cut_min: float = 0.25              # minimum histogram-change score for an abrupt-change flag
    cut_boundary_s: float = 0.4        # cuts this close to a boundary are considered explained


IDLE_WORDS = re.compile(r"\b(idle|wait|waiting|pause|paused|stationary|no action|static)\b", re.I)


def _valid(a: Annotation, duration: float, tol: float) -> bool:
    return a.start >= 0 and a.end > a.start and a.start < duration + tol


def _clamped(a: Annotation, duration: float) -> tuple[float, float]:
    return max(a.start, 0.0), min(a.end, duration)


def uncovered_intervals(annotations: list[Annotation], duration: float, tol: float = 0.05) -> list[tuple[float, float]]:
    """Intervals of [0, duration] not covered by any valid annotation."""
    spans = sorted(_clamped(a, duration) for a in annotations if _valid(a, duration, tol))
    out, cursor = [], 0.0
    for s, e in spans:
        if s > cursor + tol:
            out.append((round(cursor, 3), round(s, 3)))
        cursor = max(cursor, e)
    if duration > cursor + tol:
        out.append((round(cursor, 3), round(duration, 3)))
    return out


# ------------------------------------------------------------------------------- layer 1: timestamps
def timestamp_flags(annotations: list[Annotation], duration: float, cfg: QCConfig = QCConfig()) -> list[Flag]:
    flags: list[Flag] = []
    R = FlagOrigin.rule

    def add(code, sev, a, reason, **details):
        flags.append(Flag(code=code, category="timestamp", severity=sev, origin=R, annotation_id=a.id,
                          start=a.start, end=a.end, reason=reason, details=details))

    for a in annotations:
        if a.start < 0 or a.end < 0:
            add("TS_NEGATIVE", Severity.high, a, f"Negative timestamp ({a.start:.2f}s → {a.end:.2f}s).")
        if a.end < a.start:
            add("TS_REVERSED", Severity.high, a,
                f"End ({a.end:.2f}s) is before start ({a.start:.2f}s); interval is reversed.")
        elif a.end == a.start:
            add("TS_ZERO_LENGTH", Severity.medium, a, f"Zero-length interval at {a.start:.2f}s.")
        elif a.end - a.start < cfg.min_len_s:
            add("TS_TOO_SHORT", Severity.low, a,
                f"Very short segment ({a.end - a.start:.2f}s < {cfg.min_len_s:.2f}s).", duration=a.end - a.start)
        if max(a.start, a.end) > duration + cfg.tolerance_s:
            add("TS_OUT_OF_RANGE", Severity.high, a,
                f"Timestamp {max(a.start, a.end):.2f}s is beyond the video length ({duration:.2f}s).",
                video_duration=duration)
        if not a.action.strip():
            add("TXT_EMPTY", Severity.medium, a, "Action description is empty.")

    valid = sorted((a for a in annotations if _valid(a, duration, cfg.tolerance_s)), key=lambda a: a.start)
    for i, a in enumerate(valid):
        for b in valid[i + 1:]:
            if b.start >= a.end - cfg.tolerance_s:
                break
            ov = min(a.end, b.end) - b.start
            for x, y in ((a, b), (b, a)):
                add("TS_OVERLAP", Severity.medium, x,
                    f"Overlaps '{y.action}' ({y.start:.2f}–{y.end:.2f}s) by {ov:.2f}s. One of the two "
                    "boundaries is probably wrong.", other_id=y.id, overlap_s=round(ov, 3))

    durs = [a.end - a.start for a in valid]
    if len(durs) >= 3:
        med = float(np.median(durs))
        for a in valid:
            d = a.end - a.start
            if d > cfg.long_min_s and d > cfg.long_factor * med:
                add("TS_LONG_SEGMENT", Severity.low, a,
                    f"Segment is {d:.2f}s, {d / med:.1f}× the median segment ({med:.2f}s). It may span "
                    "more than one action.", median_s=round(med, 3))

    for s, e in uncovered_intervals(annotations, duration, cfg.tolerance_s):
        if e - s >= cfg.gap_s:
            flags.append(Flag(code="TS_GAP", category="timestamp", severity=Severity.low, origin=R,
                              start=s, end=e, reason=f"No annotation covers {s:.2f}–{e:.2f}s ({e - s:.2f}s). "
                              "This may be idle time; it is not by itself evidence of a missing action."))
    return flags


# ---------------------------------------------------------------------------- layer 2: pixel signals
def _active_mask(sig: SignalProfile) -> np.ndarray:
    return sig.motion > sig.active_threshold


def _longest_run(mask: np.ndarray, step: float) -> float:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best * step


def _cut_threshold(sig: SignalProfile, cfg: QCConfig) -> float:
    if sig.cut.size == 0:
        return float("inf")
    med = float(np.median(sig.cut))
    mad = float(np.median(np.abs(sig.cut - med))) + 1e-6
    return max(cfg.cut_min, med + 8 * mad)


def signal_flags(annotations: list[Annotation], sig: SignalProfile, duration: float,
                 cfg: QCConfig = QCConfig()) -> list[Flag]:
    flags: list[Flag] = []
    S = FlagOrigin.signal
    active = _active_mask(sig)
    thr = sig.active_threshold
    valid = [a for a in annotations if _valid(a, duration, cfg.tolerance_s)]
    covered = [_clamped(a, duration) for a in valid]

    def covered_by_other(t0: float, t1: float, me: Annotation) -> bool:
        return any(s <= t0 + cfg.tolerance_s and e >= t1 - cfg.tolerance_s
                   for (s, e), o in zip(covered, valid) if o.id != me.id)

    def frac_active(t0: float, t1: float) -> float:
        m = sig.window(t0, t1)
        return float(active[m].mean()) if m.any() else 0.0

    for a in valid:
        s, e = _clamped(a, duration)
        m = sig.window(s, e)
        if not m.any():
            continue
        frac = float(active[m].mean())
        mean_motion = float(sig.motion[m].mean())
        base = dict(active_fraction=round(frac, 3), mean_motion=round(mean_motion, 3),
                    active_threshold=round(thr, 3))
        if frac < cfg.low_activity_frac and not IDLE_WORDS.search(a.action):
            subtle = float((sig.motion[m] > sig.subtle_threshold).mean())
            base["subtle_motion_fraction"] = round(subtle, 3)
            base["subtle_threshold"] = round(sig.subtle_threshold, 3)
            if subtle >= 0.5:
                sev, extra = Severity.low, (" Only subtle motion is present, which is consistent with fine "
                                            "manipulation (e.g. a gripper closing) but not proof of it.")
            else:
                sev, extra = Severity.medium, " The frames are essentially static."
            flags.append(Flag(code="EV_LOW_MOTION", category="evidence", severity=sev, origin=S,
                              annotation_id=a.id, start=a.start, end=a.end, details=base,
                              reason=f"Only {frac:.0%} of this segment shows clear motion, yet it is described as "
                                     f"an action ('{a.action}')." + extra))
        idle = _longest_run(~active[m], sig.step)
        if idle >= cfg.idle_run_s and idle >= 0.25 * (e - s) and frac >= cfg.low_activity_frac:
            flags.append(Flag(code="EV_IDLE_INSIDE", category="boundary", severity=Severity.medium, origin=S,
                              annotation_id=a.id, start=a.start, end=a.end, details={**base, "idle_run_s": idle},
                              reason=f"Contains a {idle:.1f}s motionless stretch. The segment may include idle "
                                     "time or merge two separate actions; check its boundaries."))
        w = cfg.boundary_window_s
        if s - w >= 0 and frac_active(s - w, s - sig.step / 2) > 0.8 and not covered_by_other(s - w, s, a):
            flags.append(Flag(code="BD_START_MAY_BE_LATE", category="boundary", severity=Severity.medium,
                              origin=S, annotation_id=a.id, start=a.start, end=a.end, details=base,
                              reason=f"Motion is already ongoing in the {w:.1f}s before the start ({s:.2f}s) and "
                                     "no other annotation covers it. The start may be too late."))
        if e + w <= duration and frac_active(e + sig.step / 2, e + w) > 0.8 and not covered_by_other(e, e + w, a):
            flags.append(Flag(code="BD_END_MAY_BE_EARLY", category="boundary", severity=Severity.medium,
                              origin=S, annotation_id=a.id, start=a.start, end=a.end, details=base,
                              reason=f"Motion continues in the {w:.1f}s after the end ({e:.2f}s) and no other "
                                     "annotation covers it. The end may be too early."))

    # Abrupt visual changes (cuts, lighting jumps, occlusions) not explained by a boundary.
    cthr = _cut_threshold(sig, cfg)
    boundaries = [t for a in valid for t in _clamped(a, duration)]
    spikes = np.where(sig.cut > cthr)[0]
    last_t = -1e9
    for i in spikes:
        t = float(sig.times[i])
        if t - last_t < 1.0:
            continue
        last_t = t
        if any(abs(t - b) <= cfg.cut_boundary_s for b in boundaries):
            continue
        owner = next((a for a, (s, e) in zip(valid, covered) if s <= t <= e), None)
        flags.append(Flag(code="BD_ABRUPT_CHANGE", category="boundary", severity=Severity.low, origin=S,
                          annotation_id=owner.id if owner else None, start=round(t - sig.step, 2), end=round(t, 2),
                          details={"cut_score": round(float(sig.cut[i]), 3), "cut_threshold": round(cthr, 3)},
                          reason=f"Abrupt visual change at {t:.2f}s (histogram change {sig.cut[i]:.2f} > "
                                 f"{cthr:.2f}) that no annotation boundary explains. Could be a cut, occlusion, "
                                 "lighting change, or an unannotated event."))

    # Possible missing actions: uncovered intervals with sustained motion. Candidates only.
    for s, e in uncovered_intervals(annotations, duration, cfg.tolerance_s):
        if e - s < cfg.missing_min_s:
            continue
        frac = frac_active(s, e)
        w = sig.window(s, e)
        subtle = float((sig.motion[w] > sig.subtle_threshold).mean()) if w.any() else 0.0
        if frac >= cfg.missing_active_frac:
            sev, what = Severity.medium, f"{frac:.0%} of it shows clear motion"
        elif subtle >= cfg.missing_active_frac:
            sev, what = Severity.low, f"{subtle:.0%} of it shows subtle motion (possible fine manipulation)"
        else:
            continue
        flags.append(Flag(code="MISSING_CANDIDATE", category="missing_candidate", severity=sev,
                          origin=S, start=s, end=e,
                          details={"active_fraction": round(frac, 3), "subtle_motion_fraction": round(subtle, 3)},
                          reason=f"{s:.2f}–{e:.2f}s is not annotated but {what}. Possible missing action "
                                 "(candidate — needs human confirmation)."))
    return flags


# ------------------------------------------------------------------------------- layer 3: model judge
_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "string"},
                       "verdict": {"type": "string", "enum": ["supported", "contradicted", "insufficient"]},
                       "observed": {"type": "string"},
                       "explanation": {"type": "string"}},
        "required": ["id", "verdict", "observed", "explanation"], "additionalProperties": False}}},
    "required": ["verdicts"], "additionalProperties": False,
}

_VERIFY_PROMPT = """You are auditing action annotations of a robot manipulation video. For each annotation you
get its id, claimed time interval, claimed action, and frames sampled at its start, middle and end.
Judge whether the frames support the claimed action:
- "supported": the frames are consistent with the claimed action, including the object involved.
- "contradicted": the frames show something clearly different (wrong action, wrong object, or no action).
- "insufficient": the frames cannot settle it (e.g. the action is too brief to see in stills).
In "observed", briefly state what the robot actually appears to do. Be strict about object identity and
about whether the claimed state change (grasped / lifted / released / placed) is visible."""


def model_verify(annotations: list[Annotation], settings=None) -> dict[str, dict]:
    """Ask a vision model whether each annotation's evidence frames support its description.

    Returns {annotation_id: {"verdict", "observed", "explanation", "snapshot"}}. Raises AdapterUnavailable.
    """
    from .annotation_engine import AdapterUnavailable, _b64, claude_json_call
    from .config import get_settings
    settings = settings or get_settings()
    if not settings.has_anthropic_key:
        raise AdapterUnavailable("Set ANTHROPIC_API_KEY to enable the model-based evidence check.")
    content: list[dict] = [{"type": "text", "text": _VERIFY_PROMPT}]
    targets = [a for a in annotations if a.evidence]
    for a in targets:
        content.append({"type": "text", "text": f"\n--- id={a.id} | {a.start:.2f}s–{a.end:.2f}s | "
                                                f"claimed: \"{a.action}\""})
        for f in a.evidence:
            if f.path and Path(f.path).exists():
                content.append({"type": "text", "text": f"frame t={f.t:.2f}s"})
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                             "data": _b64(f.path)}})
    if not targets:
        return {}
    raw = claude_json_call(settings, content, _VERIFY_SCHEMA)
    known = {a.id: a for a in targets}
    out = {}
    for v in raw.get("verdicts", []):
        a = known.get(str(v.get("id")))
        if a is None or v.get("verdict") not in {"supported", "contradicted", "insufficient"}:
            continue  # ignore ids the model invented
        out[a.id] = {"verdict": v["verdict"], "observed": str(v.get("observed", ""))[:300],
                     "explanation": str(v.get("explanation", ""))[:500], "model": settings.anthropic_model,
                     "snapshot": snapshot(a)}
    return out


def snapshot(a: Annotation) -> str:
    return f"{a.start:.3f}|{a.end:.3f}|{a.action.strip()}"


def model_flags(annotations: list[Annotation], verdicts: dict[str, dict]) -> list[Flag]:
    """Turn verifier output into flags. Verdicts for since-edited annotations are not reused."""
    flags = []
    for a in annotations:
        v = verdicts.get(a.id)
        if not v:
            continue
        if v.get("snapshot") != snapshot(a):
            continue  # annotation changed after verification; verdict is stale
        if v["verdict"] == "contradicted":
            sev, code = Severity.high, "MODEL_CONTRADICTED"
        elif v["verdict"] == "insufficient":
            sev, code = Severity.low, "MODEL_INSUFFICIENT"
        else:
            continue
        flags.append(Flag(code=code, category="evidence", severity=sev, origin=FlagOrigin.model,
                          annotation_id=a.id, start=a.start, end=a.end,
                          details={k: v[k] for k in ("verdict", "observed", "model")},
                          reason=f"Vision model verdict: {v['verdict']}. Observed: {v['observed']} "
                                 f"({v['explanation']}) — model judgment, not ground truth."))
    return flags


# ------------------------------------------------------------------------------------- orchestration
def run_qc(annotations: list[Annotation], duration: float, sig: Optional[SignalProfile] = None,
           verdicts: Optional[dict[str, dict]] = None, cfg: QCConfig = QCConfig()) -> list[Flag]:
    flags = timestamp_flags(annotations, duration, cfg)
    if sig is not None and sig.motion.size:
        flags += signal_flags(annotations, sig, duration, cfg)
    if verdicts:
        flags += model_flags(annotations, verdicts)
    # A missing-action candidate already describes its gap; drop the duplicate TS_GAP for the same span.
    cand = {(f.start, f.end) for f in flags if f.code == "MISSING_CANDIDATE"}
    return [f for f in flags if not (f.code == "TS_GAP" and (f.start, f.end) in cand)]


def carry_resolutions(new: list[Flag], old: list[Flag]) -> list[Flag]:
    """Keep reviewer resolutions for flags that reappear after a re-check."""
    resolved = {f.key: f for f in old if f.resolved}
    for f in new:
        if f.key in resolved:
            f.resolved = True
            f.resolution_note = resolved[f.key].resolution_note
    return new


def priority_score(flags: list[Flag]) -> int:
    """Transparent rule: 3 per unresolved high, 2 per medium, 1 per low flag. Not a probability."""
    return sum(SEVERITY_WEIGHT[f.severity] for f in flags if not f.resolved)


PRIORITY_RULE = "Priority = 3 × high + 2 × medium + 1 × low unresolved flags (ties broken by time)."


def rank_annotations(annotations: list[Annotation], flags: list[Flag]) -> list[tuple[Annotation, int, list[Flag]]]:
    by_ann: dict[str, list[Flag]] = {}
    for f in flags:
        if f.annotation_id:
            by_ann.setdefault(f.annotation_id, []).append(f)
    rows = [(a, priority_score(by_ann.get(a.id, [])), by_ann.get(a.id, [])) for a in annotations]
    return sorted(rows, key=lambda r: (-r[1], min(r[0].start, r[0].end)))
