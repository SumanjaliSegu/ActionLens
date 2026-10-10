import json
import shutil
from pathlib import Path

import pytest

from robotrace.annotation_engine import (AdapterUnavailable, ClaudeFrameAdapter, MotionSegmentationAdapter,
                                         RemoteAdapter, SampleAdapter)
from robotrace.config import Settings
from robotrace.storage import write_in_chunks
from robotrace.video_processor import (VideoError, compute_signals, extract_frames, probe, time_to_frame,
                                       uniform_times, validate_disk_space, validate_duration, validate_upload)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "robotrace_sample.mp4"


def test_time_frame_conversion():
    assert time_to_frame(1.0, 10, 100) == 10
    assert time_to_frame(-5, 10, 100) == 0
    assert time_to_frame(99, 10, 100) == 99
    assert uniform_times(10, 5) == [1.0, 3.0, 5.0, 7.0, 9.0]


def test_upload_validation(tmp_path):
    s = Settings(max_upload_mb=1, work_dir=tmp_path)
    validate_upload("a.mp4", 1000, s)  # accepted: below the configured limit
    for name, size in (("a.exe", 10), ("a.mp4", 2 * 1024 * 1024), ("a.mp4", 0)):
        with pytest.raises(VideoError):
            validate_upload(name, size, s)


def test_upload_rejection_message_shows_configured_limit(tmp_path):
    """The limit is configurable (ROBOTRACE_MAX_UPLOAD_MB) and the rejection message must reflect whatever
    value is actually configured, not a hard-coded number."""
    for limit_mb in (1, 500, 999):
        s = Settings(max_upload_mb=limit_mb, work_dir=tmp_path)
        with pytest.raises(VideoError, match=f"limit is {limit_mb} MB"):
            validate_upload("a.mp4", int((limit_mb + 1) * 1024 * 1024), s)


def test_upload_accepted_up_to_configured_limit(tmp_path):
    s = Settings(max_upload_mb=5, work_dir=tmp_path)
    validate_upload("a.mp4", 5 * 1024 * 1024, s)       # exactly at the limit: accepted
    with pytest.raises(VideoError):
        validate_upload("a.mp4", 5 * 1024 * 1024 + 1, s)  # one byte over: rejected


def test_disk_space_check_rejects_when_insufficient(tmp_path, monkeypatch):
    s = Settings(max_upload_mb=500, work_dir=tmp_path)

    class FakeUsage:
        free = 10 * 1024 * 1024  # 10 MB free

    monkeypatch.setattr(shutil, "disk_usage", lambda _p: FakeUsage())
    with pytest.raises(VideoError, match="Not enough temporary disk space"):
        validate_disk_space(100 * 1024 * 1024, s)       # needs ~300 MB (3x safety factor), only 10 MB free


def test_disk_space_check_allows_when_sufficient(tmp_path, monkeypatch):
    s = Settings(max_upload_mb=500, work_dir=tmp_path)

    class FakeUsage:
        free = 10 * 1024 * 1024 * 1024  # 10 GB free

    monkeypatch.setattr(shutil, "disk_usage", lambda _p: FakeUsage())
    validate_disk_space(100 * 1024 * 1024, s)            # does not raise


def test_disk_space_check_does_not_block_upload_when_undeterminable(tmp_path, monkeypatch):
    """If free space can't be read (odd permissions, exotic filesystem), fail open rather than blocking
    every upload on a host where disk_usage happens not to work."""
    s = Settings(max_upload_mb=500, work_dir=tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: (_ for _ in ()).throw(OSError("no such volume")))
    validate_disk_space(100 * 1024 * 1024, s)            # does not raise


def test_write_in_chunks_reproduces_data_exactly(tmp_path):
    for size in (0, 1, (1 << 20) - 1, 1 << 20, (1 << 20) + 1, 3 * (1 << 20) + 17):
        data = bytes((i % 256 for i in range(size)))
        dest = tmp_path / f"out_{size}.bin"
        written = write_in_chunks(data, dest, chunk_size=1 << 20)
        assert written == size == dest.stat().st_size
        assert dest.read_bytes() == data


def test_write_in_chunks_accepts_a_memoryview_like_getbuffer_would_return(tmp_path):
    import io
    buf = io.BytesIO(b"x" * 2500).getbuffer()
    dest = tmp_path / "out.bin"
    assert write_in_chunks(buf, dest, chunk_size=1000) == 2500
    assert dest.read_bytes() == b"x" * 2500


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


def test_extract_frames_seeks_instead_of_rescanning_from_zero(tiny_video, tmp_path, monkeypatch):
    """Regression for the measured O(video length) per-call cost (549.52s -> 95.06s, 5.8x, on a 768s/10
    window case): a call for a late timestamp must seek near it rather than grab/read from frame 0."""
    import cv2
    meta = probe(tiny_video)
    seeks = []
    real_set = cv2.VideoCapture.set

    def tracking_set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            seeks.append(value)
        return real_set(self, prop, value)
    monkeypatch.setattr(cv2.VideoCapture, "set", tracking_set)

    extract_frames(tiny_video, meta, [2.95], tmp_path / "late")  # last frame only: nothing to seek past 0 for
    assert seeks == [29]  # time_to_frame(2.95, 10, 30) == 29


def test_extract_frames_late_frame_matches_full_sequential_scan(tiny_video, tmp_path):
    """Correctness check for the seek optimisation: a single late-frame request must return the exact
    same frame (by index and pixel content) as decoding sequentially from frame 0, not an off-by-a-few
    frame from an imprecise keyframe seek."""
    import hashlib
    meta = probe(tiny_video)
    full = extract_frames(tiny_video, meta, [0.0, 0.1, 0.2, 2.95], tmp_path / "full")  # includes frame 0
    late_only = extract_frames(tiny_video, meta, [2.95], tmp_path / "late_only")
    full_last = next(f for f in full if f.frame_index == 29)
    late_last = next(f for f in late_only if f.frame_index == 29)
    assert full_last.frame_index == late_last.frame_index == 29
    assert (hashlib.sha256(Path(full_last.path).read_bytes()).digest()
           == hashlib.sha256(Path(late_last.path).read_bytes()).digest())


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
