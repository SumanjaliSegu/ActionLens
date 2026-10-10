"""Compare annotations with human-verified reference segments, and QC flags with known error labels.

Only computed when reference data is supplied. Nothing here produces numbers without a reference.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schemas import Annotation, Flag


@dataclass
class Segment:
    start: float
    end: float
    action: str


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def normalize_label(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    s = re.sub(r"\b(the|a|an|arm|robot)\b", " ", s)
    return " ".join(s.split())


@dataclass
class SegmentEval:
    n_pred: int
    n_ref: int
    iou_threshold: float
    matches: list[dict] = field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0
    mean_iou: float = 0.0
    mean_start_error_s: float = 0.0
    mean_end_error_s: float = 0.0
    label_accuracy: float = 0.0

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def evaluate_segments(pred: list[Annotation], ref: list[Segment], iou_threshold: float = 0.5) -> SegmentEval:
    """Greedy one-to-one matching by IoU. Invalid predictions (end <= start) can never match.

    label_accuracy uses normalised exact string match (lower-case, articles removed): strict by design.
    """
    cands = []
    for i, p in enumerate(pred):
        if p.end <= p.start:
            continue
        for j, r in enumerate(ref):
            v = iou((p.start, p.end), (r.start, r.end))
            if v >= iou_threshold:
                cands.append((v, i, j))
    cands.sort(reverse=True)
    used_p, used_r, matches = set(), set(), []
    for v, i, j in cands:
        if i in used_p or j in used_r:
            continue
        used_p.add(i)
        used_r.add(j)
        p, r = pred[i], ref[j]
        matches.append({"pred_id": p.id, "pred_action": p.action, "ref_action": r.action, "iou": round(v, 3),
                        "start_error_s": round(abs(p.start - r.start), 3),
                        "end_error_s": round(abs(p.end - r.end), 3),
                        "label_match": normalize_label(p.action) == normalize_label(r.action)})
    ev = SegmentEval(n_pred=len(pred), n_ref=len(ref), iou_threshold=iou_threshold, matches=matches)
    if matches:
        n = len(matches)
        ev.mean_iou = round(sum(m["iou"] for m in matches) / n, 3)
        ev.mean_start_error_s = round(sum(m["start_error_s"] for m in matches) / n, 3)
        ev.mean_end_error_s = round(sum(m["end_error_s"] for m in matches) / n, 3)
        ev.label_accuracy = round(sum(m["label_match"] for m in matches) / n, 3)
    ev.precision = round(len(matches) / len(pred), 3) if pred else 0.0
    ev.recall = round(len(matches) / len(ref), 3) if ref else 0.0
    return ev


def evaluate_flags(flags: list[Flag], annotations: list[Annotation], erroneous_ids: set[str],
                   missing: list[Segment]) -> dict:
    """Annotation-level flag precision/recall + missing-action candidate hit rate.

    An annotation counts as 'flagged' if it has at least one flag (resolved or not) from the QC pass.
    """
    flagged = {f.annotation_id for f in flags if f.annotation_id}
    ids = {a.id for a in annotations}
    tp = len(flagged & erroneous_ids & ids)
    fp = len((flagged - erroneous_ids) & ids)
    fn = len((erroneous_ids - flagged) & ids)
    cand = [f for f in flags if f.category == "missing_candidate" or f.code == "TS_GAP"]
    hit = sum(any(f.start is not None and iou((f.start, f.end), (m.start, m.end)) > 0 or
                  (f.start is not None and f.start <= m.start and f.end >= m.end) for f in cand) for m in missing)
    per_error_caught = {i: sorted({f.code for f in flags if f.annotation_id == i}) for i in sorted(erroneous_ids & ids)}
    return {
        "flagged_annotations": len(flagged & ids), "erroneous_annotations": len(erroneous_ids & ids),
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "flag_precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "flag_recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "missing_segments": len(missing), "missing_with_candidate_or_gap_flag": hit,
        "per_error_caught": per_error_caught,
    }
