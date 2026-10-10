"""Full-frame-rate motion trimming and parallel window requests (offline; HTTP faked, no model calls)."""
import json
import re
import threading

import pytest
import requests

from robotrace import annotation_engine as ae
from robotrace.config import Settings
from robotrace.motion_refine import MotionSpan, detect_motion_span, trim_idle
from robotrace.pipeline import process_video
from robotrace.schemas import Annotation
from robotrace.video_processor import probe


def test_detects_motion_span_on_real_video(tiny_video):
    span = detect_motion_span(tiny_video)          # 10 fps: still 0-1 s, square moves 1-2 s, still after
    assert span.fps == 10 and span.frames == 30
    assert span.first_motion_s == pytest.approx(1.1, abs=0.15) and span.last_motion_s == pytest.approx(1.9, abs=0.15)


def test_trim_cuts_only_outer_idle_and_never_extends():
    span = MotionSpan(fps=10, frames=30, first_motion_s=1.1, last_motion_s=1.9)
    anns = [Annotation(start=0.0, end=1.5, action="reach"), Annotation(start=1.5, end=3.0, action="retract")]
    done = trim_idle(anns, span, 3.0)
    assert (anns[0].start, anns[1].end) == (1.0, 2.0) and anns[0].end == 1.5 and anns[1].start == 1.5
    assert done == {"start_trimmed_s": 1.0, "end_trimmed_s": 1.0}
    late = [Annotation(start=1.4, end=1.6, action="push")]          # starts after motion: untouched
    assert trim_idle(late, span, 3.0) == {"start_trimmed_s": 0.0, "end_trimmed_s": 0.0} and late[0].start == 1.4


def test_no_motion_means_no_trim():
    anns = [Annotation(start=0.0, end=3.0, action="x")]
    assert trim_idle(anns, MotionSpan(10, 30, None, None), 3.0)["start_trimmed_s"] == 0.0 and anns[0].start == 0.0


def _fake(monkeypatch, fail_on=None):
    calls, active, peak = [], [0], [0]
    lock = threading.Lock()

    class R:
        def __init__(self, status, body):
            self.status_code, self._b, self.text = status, body, json.dumps(body)

        def json(self):
            return self._b

    def post(url, json=None, headers=None, timeout=None):
        with lock:
            calls.append(json)
            i = len(calls)
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        import time
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        ts = [float(m) for c in json["input"][0]["content"] if c["type"] == "input_text"
              for m in re.findall(r"^t = ([0-9.]+)s$", c["text"])]
        if fail_on is not None and ts[0] > fail_on:
            return R(429, {"error": {"message": "limit"}})
        ans = {"scene_summary": "s", "robot_visible": True, "actions": [
            {"first_frame_t": ts[0], "last_frame_t": ts[-1], "action": "push the red square", "objects": ["square"],
             "visual_evidence": "moves", "evidence_frame_times": [ts[0]], "confidence": 0.8}]}
        return R(200, {"id": f"r{i}", "model": "m", "status": "completed", "usage": {"input_tokens": 1, "output_tokens": 1},
                       "output": [{"type": "message", "content": [{"type": "output_text",
                                                                   "text": __import__("json").dumps(ans)}]}]})
    monkeypatch.setattr(requests, "post", post)
    return calls, peak


@pytest.fixture
def windowed(monkeypatch):
    for k, v in {"OPENCODE_GO_API_KEY": "test-key-not-real", "ROBOTRACE_SAMPLING": "windowed", "ROBOTRACE_MAX_FRAMES": "4",
                 "ROBOTRACE_FRAME_SPACING_S": "0.25", "ROBOTRACE_WINDOW_OVERLAP_S": "0.25", "ROBOTRACE_MAX_WINDOWS": "6",
                 "ROBOTRACE_PARALLEL_REQUESTS": "3"}.items():
        monkeypatch.setenv(k, v)


def test_parallel_windows_same_result_as_sequential(windowed, monkeypatch, tiny_video):
    calls, peak = _fake(monkeypatch)
    par = process_video(tiny_video, probe(tiny_video), "muse", settings=Settings())
    assert peak[0] > 1                                               # requests really overlapped
    monkeypatch.setenv("ROBOTRACE_PARALLEL_REQUESTS", "1")
    _fake(monkeypatch)
    seq = process_video(tiny_video, probe(tiny_video), "muse", settings=Settings())
    strip = lambda s: [(a.start, a.end, a.action) for a in s.annotations]
    assert strip(par) == strip(seq) and par.processing.sampling["requests"] == seq.processing.sampling["requests"]


def test_parallel_window_failure_fails_whole_run(windowed, monkeypatch, tiny_video):
    _fake(monkeypatch, fail_on=1.0)
    with pytest.raises(ae.AdapterUnavailable, match="rate limit"):
        process_video(tiny_video, probe(tiny_video), "muse", settings=Settings())


def test_pipeline_motion_trim_records_what_it_did(windowed, monkeypatch, tiny_video):
    monkeypatch.setenv("ROBOTRACE_MOTION_TRIM", "1")
    _fake(monkeypatch)
    sess = process_video(tiny_video, probe(tiny_video), "muse", settings=Settings())
    note = next(w for w in sess.processing.warnings if w.startswith("Full-frame-rate motion analysis"))
    assert "10 fps" in note and sess.annotations[0].start >= 0.9 and sess.annotations[-1].end <= 2.1


def test_motion_trim_off_by_default(monkeypatch):
    monkeypatch.delenv("ROBOTRACE_MOTION_TRIM", raising=False)
    monkeypatch.delenv("ROBOTRACE_PARALLEL_REQUESTS", raising=False)
    s = Settings()
    assert s.motion_trim is False and s.parallel_requests == 1


def test_human_hands_video_is_annotated_and_window_details_recorded(monkeypatch, tiny_video):
    """obs-v4 regression: a human-hands demonstration (robot_visible=true for hands) yields annotations, and each
    window's own answer is kept in the run record for diagnosis."""
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-not-real")
    monkeypatch.delenv("ROBOTRACE_SAMPLING", raising=False)
    seen = []

    class R:
        status_code = 200

        def __init__(self, body):
            self._b, self.text = body, json.dumps(body)

        def json(self):
            return self._b

    def post(url, json=None, headers=None, timeout=None):
        seen.append(json["input"][0]["content"][0]["text"])
        ts = [float(m) for c in json["input"][0]["content"] if c["type"] == "input_text"
              for m in re.findall(r"^t = ([0-9.]+)s$", c["text"])]
        ans = {"scene_summary": "First-person view of human hands assembling a frame.", "robot_visible": True,
               "actions": [{"first_frame_t": ts[0], "last_frame_t": ts[-1], "action": "right hand screws the bracket",
                            "objects": ["screwdriver", "bracket"], "visual_evidence": "screwdriver turns",
                            "evidence_frame_times": [ts[0]], "confidence": 0.7}]}
        return R({"id": "r1", "model": "m", "status": "completed", "usage": {"input_tokens": 1, "output_tokens": 1},
                  "output": [{"type": "message", "content": [{"type": "output_text",
                                                              "text": __import__("json").dumps(ans)}]}]})
    monkeypatch.setattr(requests, "post", post)
    sess = process_video(tiny_video, probe(tiny_video), "muse", settings=Settings())
    assert "human hands" in seen[0] and [a.action for a in sess.annotations] == ["right hand screws the bracket"]
    d = sess.processing.sampling["window_details"]
    assert len(d) == 1 and d[0]["manipulator_visible"] is True and d[0]["actions_returned"] == 1
    assert d[0]["response_id"] == "r1" and "human hands" in d[0]["scene_summary"]
