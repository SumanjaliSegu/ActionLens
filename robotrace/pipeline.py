"""End-to-end orchestration: video -> annotations -> evidence -> QC -> ReviewSession.

Used by the Streamlit app and the CLI (`python -m robotrace.cli`). Contains no UI code.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from .annotation_engine import AdapterUnavailable, attach_evidence, get_adapter
from .config import SAMPLE_DIR, Settings, get_settings
from .quality_engine import model_verify, run_qc
from .schemas import ProcessingInfo, ReviewSession, VideoMetadata
from .video_processor import compute_signals, probe, validate_duration

SAMPLE_VIDEO = SAMPLE_DIR / "robotrace_sample.mp4"

Progress = Callable[[float, str], None]


def _noop(_p: float, _m: str) -> None:
    pass


def cache_dir_for(meta: VideoMetadata, settings: Settings) -> Path:
    d = settings.work_dir / "cache" / meta.sha256[:16]
    d.mkdir(parents=True, exist_ok=True)
    return d


def sample_metadata() -> VideoMetadata:
    meta = probe(SAMPLE_VIDEO)
    meta.is_synthetic_sample = True
    return meta


def process_video(path: Path, meta: VideoMetadata, adapter_name: str, *, model_check: bool = False,
                  settings: Optional[Settings] = None, progress: Progress = _noop) -> ReviewSession:
    settings = settings or get_settings()
    validate_duration(meta, settings)
    cache = cache_dir_for(meta, settings)
    adapter = get_adapter(adapter_name)

    progress(0.05, "Computing motion and scene-change signals…")
    t0 = time.perf_counter()
    sig = compute_signals(path, meta)
    sig_seconds = time.perf_counter() - t0

    progress(0.25, f"Generating annotations with '{adapter.label}'…")
    t1 = time.perf_counter()
    result = adapter.annotate(path, meta, cache)
    ann_seconds = time.perf_counter() - t1

    progress(0.55, "Extracting evidence frames…")
    n_frames = attach_evidence(result.annotations, path, meta, cache)

    warnings = list(result.warnings)
    verdicts: dict = {}
    t2 = time.perf_counter()
    if model_check:
        progress(0.65, "Asking the vision model to verify each annotation against its frames…")
        try:
            verdicts = model_verify(result.annotations, settings)
        except AdapterUnavailable as e:
            warnings.append(f"Model-based evidence check skipped: {e}")
    progress(0.9, "Running quality-control rules…")
    flags = run_qc(result.annotations, meta.duration_s, sig, verdicts)
    qc_seconds = time.perf_counter() - t2 + sig_seconds

    processing = ProcessingInfo(adapter=result.adapter, adapter_detail=result.adapter_detail,
                                is_live_inference=result.is_live_inference,
                                is_precomputed_sample=result.is_precomputed_sample,
                                annotation_seconds=round(ann_seconds, 3), qc_seconds=round(qc_seconds, 3),
                                sampled_frames=n_frames, warnings=warnings,
                                model_summary=result.model_summary, provider=result.provider,
                                model=result.model, prompt_version=result.prompt_version,
                                sampling=result.sampling)
    baseline = [a.model_copy(deep=True) for a in result.annotations]
    session = ReviewSession(video=meta, processing=processing, baseline=baseline,
                            annotations=[a.model_copy(deep=True) for a in result.annotations],
                            initial_flags=[f.model_copy(deep=True) for f in flags], flags=flags,
                            signals=sig.to_dict(), model_verdicts=verdicts)
    progress(1.0, "Done")
    return session


__all__ = ["process_video", "sample_metadata", "SAMPLE_VIDEO", "AdapterUnavailable", "cache_dir_for"]
