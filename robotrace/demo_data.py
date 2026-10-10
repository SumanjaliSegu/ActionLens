"""Read-only access to the cached development-benchmark artifacts for the demo UI.

Nothing here calls a model or writes to the benchmark folders. Cached annotations are the recorded output of the
approved 8-call Muse baseline run (prompt obs-v3, legacy sampling); they are shown as CACHED, never as live output.
All summary numbers are computed from the benchmark files at load time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .config import PROJECT_ROOT, Settings
from .schemas import Annotation, ProcessingInfo, ReviewSession

BENCH = PROJECT_ROOT / "out" / "bench_annotation_20261010-213141"
PHASE_ORDER = ["approach", "grasp", "transport", "release", "return"]
CACHED_ADAPTER = "muse-cached"


def _first(pattern: str) -> Optional[Path]:
    hits = sorted(BENCH.glob(pattern)) if BENCH.is_dir() else []
    return hits[0] if hits else None


def _baseline() -> Optional[dict]:
    p = _first("baseline_obs-v3_legacy_*/baseline.json")
    return json.loads(p.read_text(encoding="utf-8")) if p else None


def available() -> bool:
    return _baseline() is not None


def cached_clips() -> list[dict]:
    """The 8 baseline clips whose cached prediction and clip file both exist."""
    b = _baseline()
    out = []
    for r in (b or {}).get("results", []):
        clip = PROJECT_ROOT / r["clip"]
        if r.get("status") == "ok" and clip.exists():
            out.append({"key": f"{r['dataset']}_ep{r['episode']:03d}", "dataset": r["dataset"], "episode": r["episode"],
                        "split": r["split"], "task": r["task"].strip(), "clip": clip, "n_segments": len(r["annotations"]),
                        "response_id": r.get("response_id"), "model": r.get("model")})
    return out


def load_cached_session(key: str, settings: Settings) -> tuple[ReviewSession, Path]:
    """Build a review session from a cached baseline prediction (no model call). QC is re-run locally."""
    from .annotation_engine import attach_evidence
    from .pipeline import cache_dir_for
    from .quality_engine import run_qc
    from .video_processor import compute_signals, probe
    b = _baseline()
    r = next(x for x in b["results"] if f"{x['dataset']}_ep{x['episode']:03d}" == key)
    clip = PROJECT_ROOT / r["clip"]
    meta = probe(clip, display_name=f"{key} ({r['task'].strip()})"[:120])
    anns = [Annotation(start=a["start"], end=a["end"], action=a["action"], objects=a.get("objects", []),
                       confidence=a.get("confidence"), confidence_kind="model_self_reported", source=CACHED_ADAPTER,
                       visual_evidence=a.get("visual_evidence", ""), cited_frame_times=a.get("cited_frame_times", []),
                       boundary_precision_s=a.get("boundary_precision_s")) for a in r["annotations"]]
    cache = cache_dir_for(meta, settings)
    n_frames = attach_evidence(anns, clip, meta, cache)
    sig = compute_signals(clip, meta)
    flags = run_qc(anns, meta.duration_s, sig)
    proc = ProcessingInfo(
        adapter=CACHED_ADAPTER, is_live_inference=False, provider=r.get("provider", ""), model=r.get("model", ""),
        prompt_version=r.get("prompt_version", ""), sampling=r.get("sampling") or {}, sampled_frames=n_frames,
        annotation_seconds=r.get("annotation_seconds") or 0.0, model_summary=r.get("model_summary", ""),
        adapter_detail=(f"CACHED output of the recorded baseline Muse run ({r.get('model')}, response "
                        f"{r.get('response_id')}); not re-run in this session."),
        warnings=["Cached development-benchmark result: these annotations were produced by a recorded live Muse "
                  "call and are replayed here without calling the model. QC flags were recomputed locally."]
                 + [w for w in r.get("warnings", []) if not w.startswith("Data use")])
    sess = ReviewSession(video=meta, processing=proc, baseline=[a.model_copy(deep=True) for a in anns],
                         annotations=[a.model_copy(deep=True) for a in anns],
                         initial_flags=[f.model_copy(deep=True) for f in flags], flags=flags, signals=sig.to_dict())
    return sess, clip


def results_summary() -> Optional[dict]:
    """Benchmark facts computed from the files (None if the files are missing)."""
    b = _baseline()
    analysis_p = _first("baseline_obs-v3_legacy_*/baseline_analysis.json")
    if not b or not analysis_p:
        return None
    rows = json.loads(analysis_p.read_text(encoding="utf-8"))["rows"]
    per = {}
    for ds in ("ur5", "so101"):
        rs = [r for r in rows if r["dataset"] == ds]
        full = [r for r in rs if r["phase_sequence"] == PHASE_ORDER]
        per[ds] = {"clips": len(rs), "five_ordered_steps": len(full),
                   "missing_or_merged": len(rs) - len(full),
                   "missing_detail": {f"ep{r['episode']}": [p for p in PHASE_ORDER if p not in r["phase_sequence"]]
                                      for r in rs if r not in full}}
    res = b["results"]
    out = {"clips": len(res), "per_dataset": per, "model": b.get("model"), "requests": b.get("requests_made"),
           "tokens_in": sum(r.get("tokens_in") or 0 for r in res), "tokens_out": sum(r.get("tokens_out") or 0 for r in res),
           "latency_s": round(sum(r.get("annotation_seconds") or 0 for r in res), 1), "exp2": None}
    e = _first("exp2_motion_trim_offline_*/results.json")
    if e:
        agg = json.loads(e.read_text(encoding="utf-8")).get("aggregates", {}).get("B_sensitive")
        if agg:
            out["exp2"] = {
                "idle_before_s": round(sum(agg[d]["idle_inside_total"][0] for d in agg), 2),
                "idle_after_s": round(sum(agg[d]["idle_inside_total"][1] for d in agg), 2),
                "wrong_cuts_s": round(sum(agg[d]["wrong_cut_total"] for d in agg), 3),
                "settling_removed_s": round(sum(agg[d]["uncertain_settling_removed_total"] for d in agg), 2),
                "per_dataset": agg}
    return out
