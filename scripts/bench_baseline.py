"""Run the approved baseline: current engine (prompt obs-v3, legacy sampling), one Muse call per listed clip.

    python scripts/bench_baseline.py --bench out/bench_annotation_<stamp> --live

Hard limits: exactly the clips in BASELINE_CLIPS, at most one HTTP request each (MAX_REQUESTS total), no retries.
Writes only into a fresh subdirectory <bench>/baseline_obs-v3_legacy_<stamp>/ (refuses if it exists). Saves the raw
response body (no request headers, no credentials) and the parsed annotations per clip, then scores the
signal-anchored timing references. Draft semantic labels are NOT scored (they need human verification).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("eval_so101", ROOT / "scripts" / "eval_so101.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

BASELINE_CLIPS = [("ur5", 4), ("ur5", 402), ("ur5", 533), ("ur5", 546), ("so101", 1), ("so101", 3), ("so101", 4),
                  ("so101", 20)]
MAX_REQUESTS = len(BASELINE_CLIPS)
EXPECTED = {"prompt_version": "obs-v3", "sampling": {"mode": "legacy", "max_frames": 24}}


class RequestRecorder:
    """Wraps requests.post: enforces the request cap and keeps the raw response body (never headers)."""

    def __init__(self, real_post, cap: int):
        self.real_post, self.cap, self.count, self.last = real_post, cap, 0, None

    def __call__(self, url, json=None, headers=None, timeout=None, **kw):
        if "opencode.ai" not in url:
            raise RuntimeError(f"unexpected request target {url!r}; refusing")
        if self.count >= self.cap:
            raise RuntimeError(f"request cap {self.cap} reached; refusing an extra live call")
        self.count += 1
        content = (json or {}).get("input", [{}])[0].get("content", [])
        summary = {"url": url, "model": (json or {}).get("model"),
                   "n_images": sum(1 for c in content if c.get("type") == "input_image"),
                   "frame_labels": [c["text"] for c in content if c.get("type") == "input_text"
                                    and c.get("text", "").startswith("t = ")]}
        t0 = time.perf_counter()
        r = self.real_post(url, json=json, headers=headers, timeout=timeout, **kw)
        self.last = {"request": summary, "http_status": r.status_code, "elapsed_s": round(time.perf_counter() - t0, 2),
                     "body": r.text}
        return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args(argv)
    bench_dir = args.bench.resolve()
    bench = json.loads((bench_dir / "benchmark.json").read_text(encoding="utf-8"))
    entries = {(e["dataset"], e["episode"]): e for e in bench["episodes"]}
    missing = [c for c in BASELINE_CLIPS if c not in entries]
    if missing:
        print(f"clips not in benchmark: {missing}", file=sys.stderr)
        return 2

    import requests
    from robotrace.annotation_engine import OBSERVATION_PROMPT_VERSION, AdapterUnavailable
    from robotrace.config import get_settings
    from robotrace.pipeline import process_video
    from robotrace.sampling import sampling_config
    from robotrace.video_processor import probe
    settings = get_settings()
    actual = {"prompt_version": OBSERVATION_PROMPT_VERSION, "sampling": sampling_config(settings)}
    if actual != EXPECTED:
        print(f"engine config {actual} != approved {EXPECTED}; refusing", file=sys.stderr)
        return 2
    if not args.live:
        print("dry run: would make", MAX_REQUESTS, "calls with", actual, "model", settings.muse_model)
        return 0
    if not settings.has_opencode_go_key:
        print("OPENCODE_GO_API_KEY not configured; refusing", file=sys.stderr)
        return 2

    out = bench_dir / f"baseline_obs-v3_legacy_{time.strftime('%Y%m%d-%H%M%S')}"
    if out.exists():
        print(f"{out} exists; refusing to overwrite", file=sys.stderr)
        return 2
    (out / "raw").mkdir(parents=True)
    (out / "predictions").mkdir()
    rec = RequestRecorder(requests.post, MAX_REQUESTS)
    requests.post = rec
    results = []
    for ds, ep in BASELINE_CLIPS:
        e = entries[(ds, ep)]
        clip = ROOT / e["clip"]
        before = rec.count
        t0 = time.perf_counter()
        item = {"dataset": ds, "episode": ep, "split": e["split"], "task": e["task"], "clip": e["clip"]}
        try:
            sess = process_video(clip, probe(clip, display_name=clip.name), "muse", settings=settings)
            p = sess.processing
            item.update(status="ok", provider=p.provider, model=p.model, prompt_version=p.prompt_version,
                        sampling=p.sampling, adapter_detail=p.adapter_detail, annotation_seconds=p.annotation_seconds,
                        model_summary=p.model_summary, warnings=p.warnings,
                        annotations=[{"start": a.start, "end": a.end, "action": a.action, "objects": a.objects,
                                      "confidence": a.confidence, "visual_evidence": a.visual_evidence,
                                      "cited_frame_times": a.cited_frame_times,
                                      "boundary_precision_s": a.boundary_precision_s} for a in sess.annotations],
                        qc_flags=[{"code": f.code, "severity": f.severity.value, "start": f.start, "end": f.end}
                                  for f in sess.flags if not f.resolved])
            m = ev._DETAIL_RE.search(p.adapter_detail or "")
            item.update(response_id=m.group("rid") if m else None,
                        tokens_in=int(m.group("tin")) if m and m.group("tin") else None,
                        tokens_out=int(m.group("tout")) if m and m.group("tout") else None)
        except (AdapterUnavailable, Exception) as ex:  # noqa: BLE001 - every failure is recorded, never hidden
            item.update(status="failed", error=f"{type(ex).__name__}: {ex}")
        item["requests_made"] = rec.count - before
        item["wall_seconds"] = round(time.perf_counter() - t0, 2)
        if rec.last and rec.count > before:
            (out / "raw" / f"{ds}_ep{ep:03d}_response.json").write_text(json.dumps(rec.last, indent=2), encoding="utf-8")
        (out / "predictions" / f"{ds}_ep{ep:03d}.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
        if item["status"] == "ok":
            item["metrics"] = ev.score_episode(e["reference_for_scoring"], item["annotations"])
        results.append(item)
        print(f"{ds} ep{ep}: {item['status']} requests={item['requests_made']} "
              f"{item.get('tokens_in')}/{item.get('tokens_out')} tokens {item.get('error', '')}")
    summary = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "engine": actual, "model": settings.muse_model,
               "requests_made": rec.count, "request_cap": MAX_REQUESTS, "mapping_version": ev.MAPPING_VERSION,
               "label_status": "timing scored against signal events only; semantic labels unverified",
               "results": results}
    (out / "baseline.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nrequests made: {rec.count} / cap {MAX_REQUESTS}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
