import json
from pathlib import Path

import pytest

from robotrace.annotation_engine import (AdapterUnavailable, ClaudeFrameAdapter, MotionSegmentationAdapter,
                                         RemoteAdapter, SampleAdapter)
from robotrace.config import Settings
from robotrace.video_processor import (VideoError, compute_signals, extract_frames, probe, time_to_frame,
                                       uniform_times, validate_duration, validate_upload)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "robotrace_sample.mp4"


def test_time_frame_conversion():
    assert time_to_frame(1.0, 10, 100) == 10
    assert time_to_frame(-5, 10, 100) == 0
    assert time_to_frame(99, 10, 100) == 99
    assert uniform_times(10, 5) == [1.0, 3.0, 5.0, 7.0, 9.0]


def test_upload_validation():
    s = Settings(max_upload_mb=1)
    validate_upload("a.mp4", 1000, s)
    for name, size in (("a.exe", 10), ("a.mp4", 2 * 1024 * 1024), ("a.mp4", 0)):
        with pytest.raises(VideoError):
            validate_upload(name, size, s)


def test_probe_and_duration_limit(tiny_video):
    meta = probe(tiny_video)
    assert meta.frame_count == 30 and meta.fps == 10 and meta.duration_s == 3.0 and len(meta.sha256) == 64
    with pytest.raises(VideoError):
        validate_duration(meta, Settings(max_duration_s=1))


def test_corrupt_video_rejected(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video at all" * 100)
    with pytest.raises(VideoError):
        probe(bad)


def test_signals_and_motion_adapter(tiny_video, tmp_path):
    meta = probe(tiny_video)
    sig = compute_signals(tiny_video, meta, hz=10)
    assert sig.mean_motion(1.1, 1.9) > sig.mean_motion(2.3, 2.9)
    res = MotionSegmentationAdapter(min_len_s=0.3).annotate(tiny_video, meta, tmp_path)
    assert len(res.annotations) == 1
    a = res.annotations[0]
    assert 0.8 <= a.start <= 1.3 and 1.7 <= a.end <= 2.2 and "unlabeled" in a.action


def test_extract_frames(tiny_video, tmp_path):
    meta = probe(tiny_video)
    frames = extract_frames(tiny_video, meta, [0.0, 1.5, 2.95], tmp_path / "f")
    assert [f.frame_index for f in frames] == [0, 15, 29]
    assert all(Path(f.path).exists() for f in frames)


def test_sample_adapter_refuses_other_videos(tiny_video, tmp_path):
    with pytest.raises(AdapterUnavailable):
        SampleAdapter().annotate(tiny_video, probe(tiny_video), tmp_path)


def test_live_adapters_report_unavailable_without_config(monkeypatch, tiny_video, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ROBOTRACE_REMOTE_URL", raising=False)
    c = ClaudeFrameAdapter(Settings())
    assert c.availability()[0] is False
    with pytest.raises(AdapterUnavailable):
        c.annotate(tiny_video, probe(tiny_video), tmp_path)
    r = RemoteAdapter(Settings())
    assert r.availability()[0] is False
    with pytest.raises(AdapterUnavailable):
        r.annotate(tiny_video, probe(tiny_video), tmp_path)


def test_remote_adapter_unreachable(monkeypatch, tiny_video, tmp_path):
    monkeypatch.setenv("ROBOTRACE_REMOTE_URL", "http://127.0.0.1:9")  # discard port: nothing listens
    r = RemoteAdapter(Settings())
    ok, why = r.health()
    assert not ok and "unreachable" in why


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample not generated")
def test_end_to_end_sample(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTRACE_WORK_DIR", str(tmp_path))
    from robotrace import export_service as ex
    from robotrace.pipeline import process_video, sample_metadata
    meta = sample_metadata()
    sess = process_video(SAMPLE, meta, "sample", settings=Settings())
    assert sess.processing.is_precomputed_sample and not sess.processing.is_live_inference
    assert len(sess.annotations) == 11 and all(a.evidence for a in sess.annotations if a.end > a.start)
    codes = {f.code for f in sess.flags}
    assert {"TS_REVERSED", "TS_OUT_OF_RANGE", "TS_OVERLAP", "EV_LOW_MOTION", "MISSING_CANDIDATE"} <= codes
    base = json.loads((ROOT / "data/sample/baseline_annotations.json").read_text())
    erroneous = {a["id"] for a in base["annotations"] if a["injected_error"]}
    flagged = {f.annotation_id for f in sess.flags if f.annotation_id}
    assert erroneous <= flagged
    json.loads(ex.review_package_json(sess))


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample not generated")
def test_motion_adapter_on_sample(tmp_path):
    from robotrace.pipeline import process_video, sample_metadata
    sess = process_video(SAMPLE, sample_metadata(), "motion", settings=Settings(work_dir=tmp_path))
    assert sess.processing.is_live_inference and not sess.processing.is_precomputed_sample
    assert len(sess.annotations) >= 3
