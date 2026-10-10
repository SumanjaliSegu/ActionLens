"""Run one video through the full pipeline (engine -> evidence frames -> QC) without the UI.

    python scripts/run_inference.py path/to/video.mp4 --engine muse --out out/review_package.json

Prints which engine ran, the model's observations and the QC flags, and writes the review package JSON.
Exits non-zero (and writes nothing) if the engine fails; there is no fallback to another engine.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robotrace.annotation_engine import ADAPTERS, AdapterUnavailable  # noqa: E402
from robotrace.config import get_settings  # noqa: E402
from robotrace.export_service import review_package_json  # noqa: E402
from robotrace.pipeline import process_video  # noqa: E402
from robotrace.video_processor import VideoError, probe  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("--engine", default="muse", choices=sorted(ADAPTERS))
    ap.add_argument("--out", type=Path, default=Path("out/review_package.json"))
    args = ap.parse_args()

    try:
        meta = probe(args.video, display_name=args.video.name)
        sess = process_video(args.video, meta, args.engine, settings=get_settings(),
                             progress=lambda p, m: print(f"[{p:4.0%}] {m}", file=sys.stderr))
    except (AdapterUnavailable, VideoError) as e:
        print(f"FAILED ({type(e).__name__}): {e}", file=sys.stderr)
        return 1

    p = sess.processing
    print(f"\nEngine: {p.adapter} | live inference: {p.is_live_inference} | precomputed: {p.is_precomputed_sample}")
    if p.provider:
        print(f"Provider: {p.provider} | Model: {p.model}")
    print(f"Detail: {p.adapter_detail}")
    print(f"Annotation time: {p.annotation_seconds:.1f}s")
    if p.model_summary:
        print(f"Scene: {p.model_summary}")
    for w in p.warnings:
        print(f"Warning: {w}")
    print(f"\n{len(sess.annotations)} annotation(s):")
    for a in sess.annotations:
        prec = f" ±{a.boundary_precision_s:.2f}" if a.boundary_precision_s is not None else ""
        print(f"  {a.start:6.2f}–{a.end:6.2f}s{prec}  {a.action}")
        if a.objects:
            print(f"      objects: {', '.join(a.objects)}")
        if a.visual_evidence:
            print(f"      evidence: {a.visual_evidence}  (frames {', '.join(f'{t:.2f}' for t in a.cited_frame_times)})")
    print(f"\n{len(sess.flags)} QC flag(s):")
    for f in sess.flags:
        print(f"  [{f.severity.value:6}] {f.code:22} {f.reason}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(review_package_json(sess), encoding="utf-8")
    print(f"\nReview package written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
