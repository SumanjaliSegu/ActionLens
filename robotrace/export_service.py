"""Exports with a documented, versioned schema (see docs/SCHEMA.md). All times are in seconds.

Server-local evidence file paths are stripped; evidence is exported as {t, frame_index}.
"""
from __future__ import annotations

import csv
import io
import json

from .quality_engine import PRIORITY_RULE, rank_annotations
from .review_service import summary
from .schemas import SCHEMA_VERSION, TIME_UNIT, Annotation, ReviewSession, utcnow


def _ann_dict(a: Annotation, session: ReviewSession | None = None) -> dict:
    d = a.model_dump(mode="json")
    d["evidence"] = [{"t": e.t, "frame_index": e.frame_index} for e in a.evidence]
    if session is not None:
        d["review_status"] = session.status.get(a.id, "unreviewed")
        if hasattr(d["review_status"], "value"):
            d["review_status"] = d["review_status"].value
        d["reviewer_note"] = session.notes.get(a.id, "")
    return d


def _header(session: ReviewSession, kind: str) -> dict:
    return {"schema_version": SCHEMA_VERSION, "export_kind": kind, "exported_at": utcnow(),
            "time_unit": TIME_UNIT, "video": session.video.model_dump(mode="json"),
            "processing": session.processing.model_dump(mode="json")}


def annotations_json(session: ReviewSession) -> str:
    """Current (reviewed) annotations with review status."""
    body = _header(session, "annotations")
    body["annotations"] = [_ann_dict(a, session) for a in session.annotations]
    return json.dumps(body, indent=2)


CSV_FIELDS = ["id", "start_s", "end_s", "duration_s", "action", "source", "confidence", "confidence_kind",
              "review_status", "reviewer_note", "open_flags", "flag_codes", "evidence_times_s"]


def annotations_csv(session: ReviewSession) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
    w.writeheader()
    for a in session.annotations:
        flags = [f for f in session.flags if f.annotation_id == a.id and not f.resolved]
        st = session.status.get(a.id, "unreviewed")
        w.writerow({"id": a.id, "start_s": a.start, "end_s": a.end, "duration_s": round(a.end - a.start, 3),
                    "action": _csv_safe(a.action), "source": a.source,
                    "confidence": "" if a.confidence is None else a.confidence,
                    "confidence_kind": a.confidence_kind, "review_status": getattr(st, "value", st),
                    "reviewer_note": _csv_safe(session.notes.get(a.id, "")), "open_flags": len(flags),
                    "flag_codes": ";".join(sorted({f.code for f in flags})),
                    "evidence_times_s": ";".join(f"{e.t:.2f}" for e in a.evidence)})
    return buf.getvalue()


def _csv_safe(text: str) -> str:
    """Neutralise spreadsheet formula injection from untrusted model/reviewer text."""
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


def review_package_json(session: ReviewSession) -> str:
    """Everything: baseline, reviewed annotations, initial and current flags, decisions, event log."""
    body = _header(session, "review_package")
    body.update({
        "baseline_annotations": [_ann_dict(a) for a in session.baseline],
        "reviewed_annotations": [_ann_dict(a, session) for a in session.annotations],
        "deleted_annotation_ids": session.deleted_ids,
        "initial_flags": [f.model_dump(mode="json") for f in session.initial_flags],
        "current_flags": [f.model_dump(mode="json") for f in session.flags],
        "model_verdicts": {k: {kk: vv for kk, vv in v.items() if kk != "snapshot"}
                           for k, v in session.model_verdicts.items()},
        "review_events": [e.model_dump(mode="json") for e in session.events],
        "summary": summary(session),
    })
    return json.dumps(body, indent=2)


def qc_report_markdown(session: ReviewSession) -> str:
    s = summary(session)
    v, p = session.video, session.processing
    mode = ("PRECOMPUTED SAMPLE (simulated annotator, not live inference)" if p.is_precomputed_sample
            else "live inference" if p.is_live_inference else "offline")
    lines = [
        "# RoboTrace Quality-Control Report", "",
        f"- Exported: {utcnow()}  |  Schema: `{SCHEMA_VERSION}`  |  Time unit: {TIME_UNIT}",
        f"- Video: `{v.filename}`, {v.duration_s:.2f}s, {v.fps:g} fps, {v.width}×{v.height}, "
        f"sha256 `{v.sha256[:16]}…`" + ("  (**synthetic sample**)" if v.is_synthetic_sample else ""),
        f"- Annotation source: `{p.adapter}` ({mode}). {p.adapter_detail}",
        f"- Processing: annotation {p.annotation_seconds:.2f}s, QC {p.qc_seconds:.2f}s, "
        f"{p.sampled_frames} evidence frames", "",
        "## Summary", "",
        "| Metric | Before review | Current |", "|---|---|---|",
        f"| Annotations | {s['annotations_baseline']} | {s['annotations_current']} |",
        f"| Flags (open) | {s['flags_initial']['open']} | {s['flags_current']['open']} |",
        f"| Flagged annotations | {s['flags_initial']['flagged_annotations']} | "
        f"{s['flags_current']['flagged_annotations']} |", "",
        f"Review status: " + ", ".join(f"{k} {n}" for k, n in s["status"].items()),
        f"  \nReviewer actions recorded: {s['review_action_total']} "
        f"({', '.join(f'{k} {n}' for k, n in s['review_actions'].items() if k != 'recheck') or 'none'})", "",
        "## Review queue", "", f"_{PRIORITY_RULE} This is a triage ordering, not a calibrated probability._", "",
        "| # | Interval (s) | Action | Status | Priority | Open flags |", "|---|---|---|---|---|---|",
    ]
    for i, (a, score, flags) in enumerate(rank_annotations(session.annotations, session.flags), 1):
        st = session.status.get(a.id, "unreviewed")
        codes = ", ".join(sorted({f.code for f in flags if not f.resolved})) or "—"
        lines.append(f"| {i} | {a.start:.2f}–{a.end:.2f} | {a.action.replace('|', '/')} | "
                     f"{getattr(st, 'value', st)} | {score} | {codes} |")
    lines += ["", "## Flags", ""]
    for f in sorted(session.flags, key=lambda f: (f.resolved, f.start if f.start is not None else 0)):
        span = f"{f.start:.2f}–{f.end:.2f}s" if f.start is not None else ""
        state = f"RESOLVED ({f.resolution_note or 'no note'})" if f.resolved else "OPEN"
        lines.append(f"- **{f.code}** [{f.severity.value}, {f.origin.value}] {span} — {f.reason} _{state}_")
    if p.warnings:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in p.warnings]
    lines += ["", "## Interpretation notes", "",
              "- `deterministic_rule` flags are arithmetic facts about the timestamps.",
              "- `visual_signal` flags come from pixel statistics (motion energy, histogram change); they "
              "indicate where to look, not what happened.",
              "- `model_judgment` flags are opinions from a vision model and can be wrong.",
              "- MISSING_CANDIDATE and TS_GAP items are candidates; they are not confirmed omissions."]
    return "\n".join(lines) + "\n"
