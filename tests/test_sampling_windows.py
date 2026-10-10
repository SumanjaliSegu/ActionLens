"""Frame-sampling plans, cross-window merging, timestamp grounding and window failure handling.

The HTTP layer is faked. These tests check sampling/merging/validation LOGIC only; they say nothing about whether
denser or windowed sampling improves a real model's annotations (that needs live evaluation).
"""
import importlib.util
import json
import re
from pathlib import Path

import pytest
import requests

from robotrace import annotation_engine as ae
from robotrace.config import Settings
from robotrace.pipeline import process_video
from robotrace.sampling import merge_window_annotations, plan_sampling, same_action, sampling_config
from robotrace.schemas import Annotation
from robotrace.video_processor import probe, uniform_times

ROOT = Path(__file__).resolve().parents[1]
LEGACY = {"mode": "legacy", "max_frames": 24}
WINDOWED = {"mode": "windowed", "max_frames": 24, "spacing_s": 0.5, "overlap_s": 1.0, "max_windows": 4}


# ------------------------------------------------------------------------------- plans
@pytest.mark.parametrize("dur", [3.0, 6.1, 7.7, 10.1, 24.6, 40.133, 180.0])
def test_default_legacy_plan_reproduces_original_frame_times(dur):
    p = plan_sampling(dur, LEGACY)
    old_n = int(min(24, max(4, round(dur))))
    assert p.n_requests == 1 and p.windows[0].times == uniform_times(dur, old_n)
    assert p.spacing_s == pytest.approx(dur / old_n)


def test_default_settings_are_legacy(monkeypatch):
    monkeypatch.delenv("ROBOTRACE_SAMPLING", raising=False)
    monkeypatch.delenv("ROBOTRACE_MAX_FRAMES", raising=False)
    assert sampling_config(Settings()) == LEGACY
    monkeypatch.setenv("ROBOTRACE_SAMPLING", "windowed")
    assert sampling_config(Settings())["mode"] == "windowed"
    monkeypatch.setenv("ROBOTRACE_SAMPLING", "bogus")
    assert sampling_config(Settings())["mode"] == "legacy"


def test_windowed_short_clip_is_one_denser_request():
    p = plan_sampling(7.7, WINDOWED)
    assert p.n_requests == 1 and len(p.windows[0].times) == 16 and p.spacing_s == pytest.approx(7.7 / 16)


@pytest.mark.parametrize("dur, k", [(24.6, 3), (40.133, 4)])
def test_windowed_long_clip_overlapping_absolute_windows(dur, k):
    p = plan_sampling(dur, WINDOWED)
    assert p.n_requests == k and not p.capped
    assert p.windows[0].start == 0 and p.windows[-1].end == dur
    for a, b in zip(p.windows, p.windows[1:]):
        assert b.start < a.end                                   # windows overlap: no uncovered seam
    for w in p.windows:
        assert all(w.start <= t < w.end for t in w.times) and w.times == sorted(w.times)
        assert len(w.times) <= 24


def test_windowed_cost_cap_widens_spacing_and_says_so():
    p = plan_sampling(180.0, WINDOWED)
    assert p.capped and p.n_requests == 4 and p.spacing_s > 0.5 and p.windows[-1].end == 180.0
    assert p.spacing_s < 180.0 / 24                             # still denser than legacy (7.5 s)


# ------------------------------------------------------------------------------- merging
def A(s, e, action, conf=0.8, objects=()):
    return Annotation(start=s, end=e, action=action, confidence=conf, objects=list(objects))


def test_same_action_across_seam_is_merged_once():
    out, n = merge_window_annotations([[A(8.0, 11.9, "carry the red brick to the box", 0.9)],
                                       [A(11.2, 14.0, "carry the red brick to the box", 0.7)]], tol=0.25)
    assert n == 1 and len(out) == 1 and (out[0].start, out[0].end) == (8.0, 14.0)
    assert out[0].confidence == 0.7                              # conservative, never invented


def test_action_inside_overlap_reported_by_both_windows_is_deduplicated():
    out, n = merge_window_annotations([[A(1, 5, "reach"), A(11.2, 11.8, "grasp the cup")],
                                       [A(11.25, 11.75, "grasp cup"), A(12, 15, "lift the cup")]], tol=0.25)
    assert n == 1 and [a.action for a in out] == ["reach", "grasp the cup", "lift the cup"]


def test_different_actions_overlapping_at_seam_are_both_kept():
    out, n = merge_window_annotations([[A(9, 11.8, "grasp the cup")], [A(11.2, 14, "lift the cup")]], tol=0.25)
    assert n == 0 and len(out) == 2                              # left for QC (TS_OVERLAP) and the reviewer


def test_repeated_action_within_one_window_is_not_merged():
    out, n = merge_window_annotations([[A(1, 2, "push the block"), A(2, 3, "push the block")], []], tol=0.25)
    assert n == 0 and len(out) == 2                              # real tasks can repeat actions


def test_same_action_text_similarity():
    assert same_action("grasp the red cup", "grasp red cup")
    assert not same_action("grasp the red cup", "place the red cup")


# ------------------------------------------------------------------------------- end to end (fake HTTP)
@pytest.fixture
def windowed_muse(monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-not-real")
    monkeypatch.setenv("ROBOTRACE_SAMPLING", "windowed")
    monkeypatch.setenv("ROBOTRACE_MAX_FRAMES", "4")
    monkeypatch.setenv("ROBOTRACE_FRAME_SPACING_S", "0.25")
    monkeypatch.setenv("ROBOTRACE_WINDOW_OVERLAP_S", "0.25")
    monkeypatch.setenv("ROBOTRACE_MAX_WINDOWS", "6")
    return Settings()


def _window_times(body):
    return [float(m) for c in body["input"][0]["content"] if c["type"] == "input_text"
            for m in re.findall(r"^t = ([0-9.]+)s$", c["text"])]


def fake_responses_api(monkeypatch, make_answer, fail_on=None):
    calls = []

    class R:
        def __init__(self, status, body):
            self.status_code, self._b = status, body
            self.text = json.dumps(body)

        def json(self):
            return self._b

    def post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        i = len(calls)
        if fail_on == i:
            return R(429, {"error": {"message": "limit"}})
        ts = _window_times(json)
        out = {"id": f"r{i}", "model": "muse-spark-1.3-contributor", "status": "completed",
               "usage": {"input_tokens": 100, "output_tokens": 10},
               "output": [{"type": "message", "content": [{"type": "output_text",
                                                           "text": __import__("json").dumps(make_answer(i, ts))}]}]}
        return R(200, out)

    monkeypatch.setattr(requests, "post", post)
    return calls


def one_action(i, ts, action="push the red square", obj="red square"):
    return {"scene_summary": "arm pushes a square", "robot_visible": True,
            "actions": [{"first_frame_t": ts[0], "last_frame_t": ts[-1], "action": action, "objects": [obj],
                         "visual_evidence": "square moves", "evidence_frame_times": [ts[0]], "confidence": 0.8}]}


def test_windowed_muse_end_to_end_merges_seams_and_reports_all_requests(windowed_muse, monkeypatch, tiny_video):
    calls = fake_responses_api(monkeypatch, one_action)
    sess = process_video(tiny_video, probe(tiny_video), "muse", settings=windowed_muse)
    p = sess.processing
    k = len(calls)
    assert k > 1 and p.sampling["requests"] == k and p.sampling["mode"] == "windowed"
    assert len(sess.annotations) == 1 and p.sampling["merged_at_seams"] == k - 1
    a = sess.annotations[0]
    assert a.start == pytest.approx(0.0, abs=0.2) and a.end == pytest.approx(3.0, abs=0.2)
    assert f"response id {'+'.join(f'r{i}' for i in range(1, k + 1))}, tokens in/out {100 * k}/{10 * k}" \
        in p.adapter_detail
    assert any(f"{k} overlapping windows" in w for w in p.warnings)
    for body in calls:                                           # every window prompt says it is partial
        assert "cover only part of the video" in body["input"][0]["content"][0]["text"]
        ts = _window_times(body)
        assert ts == sorted(ts) and 0 <= ts[0] and ts[-1] <= 3.0   # absolute timestamps

    spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts" / "eval_so101.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    m = ev._DETAIL_RE.search(p.adapter_detail)
    assert m.group("rid") == "+".join(f"r{i}" for i in range(1, k + 1)) and int(m.group("tin")) == 100 * k


def test_one_failing_window_fails_the_whole_run_without_retry(windowed_muse, monkeypatch, tiny_video):
    calls = fake_responses_api(monkeypatch, one_action, fail_on=2)
    with pytest.raises(ae.AdapterUnavailable, match="rate limit"):
        process_video(tiny_video, probe(tiny_video), "muse", settings=windowed_muse)
    assert len(calls) == 2                                       # stopped at the failure; nothing retried


def test_object_names_differing_between_windows_are_flagged_not_rewritten(windowed_muse, monkeypatch, tiny_video):
    fake_responses_api(monkeypatch, lambda i, ts: one_action(i, ts, obj="red square" if i == 1 else "orange square"))
    sess = process_video(tiny_video, probe(tiny_video), "muse", settings=windowed_muse)
    assert any("Object names differ between windows" in w for w in sess.processing.warnings)
    assert set(sess.annotations[0].objects) == {"red square", "orange square"}


def test_ungrounded_timestamps_are_rejected(monkeypatch, tiny_video):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-not-real")
    monkeypatch.delenv("ROBOTRACE_SAMPLING", raising=False)

    def answer(i, ts):
        good = one_action(i, ts)["actions"][0]
        bad = {**good, "first_frame_t": 1.5, "last_frame_t": 1.5, "action": "invented action"}
        return {"scene_summary": "x", "robot_visible": True, "actions": [good, bad]}
    fake_responses_api(monkeypatch, answer)
    sess = process_video(tiny_video, probe(tiny_video), "muse", settings=Settings())
    assert [a.action for a in sess.annotations] == ["push the red square"]
    assert sess.processing.sampling["rejected_actions"] == 1
    assert any("1 action(s) used timestamps that match no frame" in w for w in sess.processing.warnings)


def test_all_ungrounded_output_is_a_failure_not_an_empty_result(monkeypatch, tiny_video):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-not-real")
    monkeypatch.delenv("ROBOTRACE_SAMPLING", raising=False)
    fake_responses_api(monkeypatch, lambda i, ts: {"scene_summary": "x", "robot_visible": True, "actions": [
        {**one_action(i, ts)["actions"][0], "first_frame_t": 1.5, "last_frame_t": 1.6}]})
    with pytest.raises(ae.AdapterUnavailable, match="none used the timestamps"):
        process_video(tiny_video, probe(tiny_video), "muse", settings=Settings())


def test_legacy_mode_sends_one_request_with_original_frames(monkeypatch, tiny_video):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-not-real")
    monkeypatch.delenv("ROBOTRACE_SAMPLING", raising=False)
    calls = fake_responses_api(monkeypatch, one_action)
    sess = process_video(tiny_video, probe(tiny_video), "muse", settings=Settings())
    assert len(calls) == 1 and sess.processing.sampling["config"] == LEGACY
    assert len(_window_times(calls[0])) == 4                     # min(24, max(4, round(3.0))) as before
    assert "cover only part of the video" not in calls[0]["input"][0]["content"][0]["text"]


def test_eval_cache_is_keyed_by_sampling_config(tmp_path):
    spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts" / "eval_so101.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"status": "ok", "annotations": [], "prompt_version": "obs-v3",
                                "sampling": {"config": LEGACY}}), encoding="utf-8")
    kw = dict(live=False, retry_failed=False, budget={"calls": 0, "max_calls": 1}, prompt_version="obs-v3")
    assert ev.run_inference(tmp_path / "c.mp4", path, sampling=LEGACY, **kw)["status"] == "ok"
    assert ev.run_inference(tmp_path / "c.mp4", path, sampling=WINDOWED, **kw)["status"] == "stale_cache"
    assert ev.predictions_dir(tmp_path, "obs-v3", WINDOWED, "m1").name.startswith("windowed-0.5s")
    assert ev.predictions_dir(tmp_path, "obs-v3", LEGACY, "m1") == tmp_path / "predictions" / "obs-v3" / "m1"
    assert ev.predictions_dir(tmp_path, "obs-v3", WINDOWED, "m1") != ev.predictions_dir(tmp_path, "obs-v3", LEGACY, "m1")


def test_edge_convention_timestamps_are_snapped_not_rejected(monkeypatch, tiny_video):
    """A model saying first_frame_t=0.0 / last_frame_t=<duration> (frames sit inside) keeps its action,
    with the same interval the pipeline produced before grounding existed."""
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-not-real")
    monkeypatch.delenv("ROBOTRACE_SAMPLING", raising=False)
    fake_responses_api(monkeypatch, lambda i, ts: {"scene_summary": "x", "robot_visible": True, "actions": [
        {**one_action(i, ts)["actions"][0], "first_frame_t": 0.0, "last_frame_t": 3.0}]})
    sess = process_video(tiny_video, probe(tiny_video), "muse", settings=Settings())
    assert len(sess.annotations) == 1 and sess.processing.sampling["rejected_actions"] == 0
    a = sess.annotations[0]
    assert (a.start, a.end) == (0.0, 3.0)                       # identical to the pre-grounding interval


@pytest.mark.parametrize("mode", ["legacy", "windowed"])
def test_analyze_page_shows_planned_requests_before_run(tmp_path, monkeypatch, mode):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("ROBOTRACE_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-not-real")
    monkeypatch.setenv("ROBOTRACE_SAMPLING", mode)
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("no API call may be made"))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Analyze").run()
    next(r for r in at.radio if r.label == "Source").set_value("Bundled synthetic sample").run()
    next(r for r in at.radio if r.label == "Engine").set_value("muse").run()
    assert not at.exception
    planned = [c.value for c in at.caption if c.value.startswith("Planned:")]
    assert planned and f"{mode} sampling" in planned[0] and "model request" in planned[0]
