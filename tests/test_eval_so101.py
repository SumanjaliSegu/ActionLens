"""Offline tests for scripts/eval_so101.py: reference rules, phase mapping, scoring, caching and failure handling.

No network access and no API calls. Synthetic signals stand in for the dataset; the HTTP layer is faked.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import requests

from robotrace.annotation_engine import AdapterUnavailable

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_so101", ROOT / "scripts" / "eval_so101.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

FPS = ev.FPS


def synthetic_episode(hold=True, duration=8.0):
    """Idle 0-1s, approach 1-3s (gripper opens), close 3.0-3.3s, hold 3.3-5.5s, open 5.5-5.8s, return 5.8-7s."""
    t = np.arange(int(duration * FPS)) / FPS
    state = np.zeros((len(t), 6))
    action = np.zeros((len(t), 6))
    moving = ((t >= 1.0) & (t < 7.0))
    state[:, 1] = np.cumsum(np.where(moving, 2.5, 0.0))           # 75 units/s while moving
    grip_cmd = np.full(len(t), 1.0)
    grip_cmd[(t >= 1.2) & (t < 3.0)] = 20.0
    grip_cmd[(t >= 5.5) & (t < 5.8)] = 15.0
    grip = grip_cmd.copy()
    if hold:
        grip[(t >= 3.3) & (t < 5.5)] = 7.0                         # blocked open by the object
    grip[(t >= 3.0) & (t < 3.3)] = np.linspace(20, 7 if hold else 1, int(round(0.3 * FPS)))
    state[:, 5], action[:, 5] = grip, grip_cmd
    return t, state, action


# ------------------------------------------------------------------------------- reference rules
def test_reference_events_and_phases():
    ref = ev.derive_reference(*synthetic_episode())
    e, p = ref["events"], ref["phases"]
    assert e["motion_start"] == pytest.approx(1.0, abs=0.1) and e["motion_end"] == pytest.approx(7.0, abs=0.1)
    assert e["grasp"] == pytest.approx(3.3, abs=0.05) and e["release"] == pytest.approx(5.5, abs=0.05)
    assert p["approach"][0] == pytest.approx(1.0, abs=0.1) and p["grasp"][1] == e["grasp"]
    assert p["transport"] == [e["grasp"], e["release"]]
    assert p["release"][0] == e["release"] and p["return"][1] == e["motion_end"]
    starts = [p[k][0] for k in ev.PHASES]
    assert starts == sorted(starts) and ref["notes"] == []


def test_brief_twitch_is_not_motion_start():
    t, s, a = synthetic_episode()
    s[3, 1] += 3.0                                                 # one-frame jolt at t=0.1s
    assert ev.derive_reference(t, s, a)["events"]["motion_start"] == pytest.approx(1.0, abs=0.1)


def test_no_hold_means_grasp_release_not_measurable():
    ref = ev.derive_reference(*synthetic_episode(hold=False))
    assert ref["events"]["grasp"] is None and ref["events"]["release"] is None
    assert all(v is None for v in ref["phases"].values())
    assert any("not measurable" in n for n in ref["notes"])


def test_no_motion_is_reported():
    t = np.arange(90) / FPS
    ref = ev.derive_reference(t, np.zeros((90, 6)), np.zeros((90, 6)))
    assert ref["events"]["motion_start"] is None and "no joint motion above threshold" in ref["notes"]


# ------------------------------------------------------------------------------- phase mapping
@pytest.mark.parametrize("label, phase", [
    ("move gripper toward the pink lego brick", "approach"),
    ("lower the gripper to the brick", "approach"),
    ("grasp the pink lego brick", "grasp"),
    ("pick up the brick", "grasp"),
    ("lift the brick", "transport"),
    ("move the pink lego brick over the box", "transport"),
    ("place the brick into the transparent box", "release"),
    ("lower the brick into the box", "release"),
    ("open the gripper to release the brick", "release"),
    ("retract the arm to its rest position", "return"),
    ("robot remains idle", "idle"),
    ("celebrate", None),
])
def test_phase_mapping(label, phase):
    assert ev.map_phase(label) == phase


# ------------------------------------------------------------------------------- scoring
def ref_fixture():
    return ev.derive_reference(*synthetic_episode())


def ann(s, e, action):
    return {"start": s, "end": e, "action": action}


def test_perfect_prediction_scores_high():
    ref = ref_fixture()
    p = ref["phases"]
    preds = [ann(*p["approach"], "reach toward the brick"), ann(*p["grasp"], "grasp the brick"),
             ann(*p["transport"], "lift the brick"), ann(*p["release"], "release the brick"),
             ann(*p["return"], "retract the arm")]
    m = ev.score_episode(ref, preds)
    assert m["phase_coverage"] == 1.0 and m["order_fully_correct"] and m["unmapped_predictions"] == []
    assert all(v["iou"] > 0.9 for v in m["per_phase"].values())
    assert m["timing"]["grasp"]["abs_error_s"] == 0 and m["timing"]["release"]["abs_error_s"] == 0
    assert m["frame_agreement"] > 0.95 and m["segment_match_iou0.3"]["recall"] == 1.0


def test_offset_prediction_timing_errors_and_order():
    ref = ref_fixture()
    preds = [ann(0.5, 3.0, "move to brick"), ann(3.0, 4.0, "grasp brick"), ann(4.0, 6.0, "lift the brick"),
             ann(6.0, 6.5, "retract arm"), ann(6.5, 7.0, "drop the brick")]
    m = ev.score_episode(ref, preds)
    assert m["timing"]["grasp"]["abs_error_s"] == pytest.approx(0.7, abs=0.05)      # grasp end 4.0 vs 3.3
    assert m["timing"]["release"]["abs_error_s"] == pytest.approx(1.0, abs=0.05)    # release start 6.5 vs 5.5
    assert m["predicted_phase_sequence"] == ["approach", "grasp", "transport", "return", "release"]
    assert m["order_agreement"] == 0.75 and m["order_fully_correct"] is False


def test_unmapped_and_empty_predictions_are_not_invented():
    ref = ref_fixture()
    m = ev.score_episode(ref, [ann(1, 2, "celebrate")])
    assert m["unmapped_predictions"] == ["celebrate"] and m["phase_coverage"] == 0.0
    assert m["timing"]["grasp"]["measurable"] is False and m["timing"]["grasp"]["abs_error_s"] is None
    empty = ev.score_episode(ref, [])
    assert empty["phase_coverage"] == 0.0 and empty["order_agreement"] is None


def test_unmeasurable_reference_gives_none_not_zero():
    ref = ev.derive_reference(*synthetic_episode(hold=False))
    m = ev.score_episode(ref, [ann(1, 3, "grasp the brick")])
    assert m["phase_coverage"] is None and m["timing"]["grasp"]["abs_error_s"] is None


# ------------------------------------------------------------------------------- report / serialization
def fake_episode(ep, status="ok", preds=None):
    ref = ref_fixture()
    pred = {"status": status, "annotations": preds or [], "tokens_in": 100, "tokens_out": 10,
            "annotation_seconds": 1.0, "qc_flags": [], "from_cache": False}
    if status != "ok":
        pred = {"status": status, "error": "boom", "from_cache": False}
    e = {"episode": ep, "task": "pink lego brick into the transparent box", "reference": ref, "prediction": pred}
    if status == "ok":
        e["metrics"] = ev.score_episode(ref, pred["annotations"])
    return e


def test_aggregate_excludes_failures_and_states_denominator():
    p = ref_fixture()["phases"]
    eps = [fake_episode(0, preds=[ann(*p["transport"], "lift the brick")]), fake_episode(1, status="failed")]
    agg = ev.aggregate(eps)
    assert agg["episodes_requested"] == 2 and agg["episodes_evaluated"] == 1
    assert agg["phase_coverage"]["n"] == 1 and "excluded" in agg["denominator_note"]
    att = ev.needs_attention(eps)
    assert any(a["episode"] == 1 and a["reason"].startswith("failed") for a in att)


def test_report_serializes_and_markdown_has_table(tmp_path):
    eps = [fake_episode(0, preds=[ann(1, 3, "reach for brick")]), fake_episode(1, status="failed")]
    report = {"dataset": ev.REPO, "license": "apache-2.0", "camera": "side", "fps": 30, "generated_at": "now",
              "episodes_requested": 2, "reference_rules": ev.REF_RULES, "phase_rules": ev.PHASE_RULES,
              "episodes": eps, "aggregate": ev.aggregate(eps), "needs_attention": ev.needs_attention(eps),
              "limitations": ev.LIMITATIONS}
    back = json.loads(json.dumps(report, default=str))
    assert back["aggregate"]["episodes_evaluated"] == 1
    md = ev.markdown_report(report)
    assert "| 0 | ok |" in md and "| 1 | failed |" in md and "n = 1 of 2" in md
    assert "not human annotations" in md and "## Limitations" in md


# ------------------------------------------------------------------------------- inference: caching / failures
def test_cached_success_is_reused_without_calling(tmp_path):
    from robotrace.annotation_engine import OBSERVATION_PROMPT_VERSION
    path = tmp_path / "ep.json"
    path.write_text(json.dumps({"status": "ok", "annotations": [], "prompt_version": OBSERVATION_PROMPT_VERSION}),
                    encoding="utf-8")
    budget = {"calls": 0, "max_calls": 10}
    rec = ev.run_inference(tmp_path / "clip.mp4", path, live=True, retry_failed=False, budget=budget,
                           process=lambda *a, **k: pytest.fail("must not call"))
    assert rec["status"] == "ok" and rec["from_cache"] and budget["calls"] == 0


def test_cached_prediction_from_other_prompt_version_is_never_reused_as_current(tmp_path):
    path = tmp_path / "ep.json"  # legacy record: no prompt_version field => obs-v1
    path.write_text(json.dumps({"status": "ok", "annotations": [{"start": 0, "end": 1, "action": "grasp"}]}),
                    encoding="utf-8")
    rec = ev.run_inference(tmp_path / "c.mp4", path, live=False, retry_failed=False,
                           budget={"calls": 0, "max_calls": 1}, prompt_version="obs-v2")
    assert rec["status"] == "stale_cache" and "obs-v1" in rec["error"]
    same = ev.run_inference(tmp_path / "c.mp4", path, live=False, retry_failed=False,
                            budget={"calls": 0, "max_calls": 1}, prompt_version="obs-v1")
    assert same["status"] == "ok" and same["prompt_version"] == "obs-v1"


def test_predictions_dir_keeps_legacy_files_separate(tmp_path):
    assert ev.predictions_dir(tmp_path, "obs-v1") == tmp_path / "predictions"
    assert ev.predictions_dir(tmp_path, "obs-v1", model="anything") == tmp_path / "predictions"   # frozen layout
    assert (ev.predictions_dir(tmp_path, "obs-v2", model="muse-spark-1.3-contributor")
            == tmp_path / "predictions" / "obs-v2" / "muse-spark-1.3-contributor")
    assert ev.predictions_dir(tmp_path, "obs-v2", model="a/b:c").name == "a_b_c"   # model id made path-safe


def _ok_record(path, model="muse-spark-1.3-contributor", version="obs-v3"):
    path.write_text(json.dumps({"status": "ok", "annotations": [], "prompt_version": version, "model": model,
                                "sampling": {"config": {"mode": "legacy", "max_frames": 24}}}), encoding="utf-8")


def test_cached_prediction_from_another_model_is_not_reused(tmp_path):
    path = tmp_path / "p.json"
    _ok_record(path, model="muse-spark-1.2-contributor")
    kw = dict(retry_failed=False, budget={"calls": 0, "max_calls": 1}, prompt_version="obs-v3",
              sampling={"mode": "legacy", "max_frames": 24})
    assert ev.run_inference(tmp_path / "c.mp4", path, live=False, model="muse-spark-1.2-contributor", **kw)["status"] == "ok"
    rec = ev.run_inference(tmp_path / "c.mp4", path, live=False, model="muse-spark-1.3-contributor", **kw)
    assert rec["status"] == "stale_cache" and "1.2" in rec["error"]


def test_live_run_never_overwrites_a_mismatched_or_unreadable_cached_prediction(tmp_path):
    path = tmp_path / "p.json"
    _ok_record(path, model="muse-spark-1.2-contributor")
    before = path.read_bytes()
    budget = {"calls": 0, "max_calls": 5}
    rec = ev.run_inference(tmp_path / "c.mp4", path, live=True, retry_failed=False, budget=budget,
                           prompt_version="obs-v3", sampling={"mode": "legacy", "max_frames": 24},
                           model="muse-spark-1.3-contributor", process=lambda *a, **k: pytest.fail("must not call"))
    assert rec["status"] == "stale_cache" and "not overwritten" in rec["error"]
    assert path.read_bytes() == before and budget["calls"] == 0
    bad = tmp_path / "bad.json"
    bad.write_text("{corrupt", encoding="utf-8")
    rec = ev.run_inference(tmp_path / "c.mp4", bad, live=True, retry_failed=False, budget=budget,
                           process=lambda *a, **k: pytest.fail("must not call"))
    assert rec["status"] == "invalid_cache" and bad.read_text(encoding="utf-8") == "{corrupt" and budget["calls"] == 0


def test_existing_report_is_never_overwritten_by_default(tmp_path):
    assert ev.report_stem(tmp_path, "report", overwrite=False) == "report"
    (tmp_path / "report.md").write_text("original", encoding="utf-8")
    stem = ev.report_stem(tmp_path, "report", overwrite=False)
    assert stem.startswith("report_") and stem != "report"
    assert ev.report_stem(tmp_path, "report", overwrite=True) == "report"


@pytest.mark.parametrize("label", ["move away from the clear container",
                                   "move the empty gripper away from the container",
                                   "move the arm back to the start", "retreat from the box"])
def test_mapping_v2_return_regressions(label):
    assert ev.map_phase(label) == "return"


@pytest.mark.parametrize("label, phase", [("lift the brick away from the table", "transport"),
                                          ("carry the brick away", "transport"),
                                          ("move the brick away from the edge", "transport"),
                                          ("move gripper toward the brick", "approach")])
def test_mapping_v2_does_not_overmatch_return(label, phase):
    assert ev.map_phase(label) == phase


def test_cached_failure_is_not_retried_by_default(tmp_path):
    path = tmp_path / "ep.json"
    path.write_text(json.dumps({"status": "failed", "error": "HTTP 429"}), encoding="utf-8")
    budget = {"calls": 0, "max_calls": 10}
    rec = ev.run_inference(tmp_path / "clip.mp4", path, live=True, retry_failed=False, budget=budget,
                           process=lambda *a, **k: pytest.fail("must not call"))
    assert rec["status"] == "failed" and budget["calls"] == 0


def test_not_live_and_budget_guards(tmp_path):
    assert ev.run_inference(tmp_path / "c.mp4", tmp_path / "p.json", live=False, retry_failed=False,
                            budget={"calls": 0, "max_calls": 1})["status"] == "not_run"
    assert ev.run_inference(tmp_path / "c.mp4", tmp_path / "p.json", live=True, retry_failed=False,
                            budget={"calls": 1, "max_calls": 1})["status"] == "skipped_budget"
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert ev.run_inference(tmp_path / "c.mp4", tmp_path / "bad.json", live=False, retry_failed=False,
                            budget={"calls": 0, "max_calls": 1})["status"] == "invalid_cache"


def test_adapter_failure_is_recorded_as_failure(tmp_video, tmp_path):
    def boom(*a, **k):
        raise AdapterUnavailable("OpenCode Go usage or rate limit reached [HTTP 429]")
    budget = {"calls": 0, "max_calls": 10}
    path = tmp_path / "p.json"
    rec = ev.run_inference(tmp_video, path, live=True, retry_failed=False, budget=budget, process=boom)
    assert rec["status"] == "failed" and "429" in rec["error"] and budget["calls"] == 1
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "failed"
    assert "annotations" not in rec  # a failure is never an empty prediction


def test_malformed_model_response_through_real_pipeline_is_a_failure(tmp_video, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-not-real")

    class Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"status": "completed", "output": [{"type": "message", "content": [
                {"type": "output_text", "text": "the robot did things"}]}]}
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    rec = ev.run_inference(tmp_video, tmp_path / "p.json", live=True, retry_failed=False,
                           budget={"calls": 0, "max_calls": 10})
    assert rec["status"] == "failed" and "malformed JSON" in rec["error"]


def test_successful_pipeline_record_has_ids_and_tokens(tmp_video, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-not-real")
    body = {"id": "resp_x1", "model": "muse-spark-1.3-contributor", "status": "completed",
            "usage": {"input_tokens": 1234, "output_tokens": 56},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps({
                "scene_summary": "arm and brick", "robot_visible": True, "actions": [
                    {"first_frame_t": 0.4, "last_frame_t": 1.1, "action": "grasp the brick", "objects": ["brick"],
                     "visual_evidence": "gripper closes", "evidence_frame_times": [0.4], "confidence": 0.8}]})}]}]}

    class Resp:
        status_code = 200
        text = json.dumps(body)

        def json(self):
            return body
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    rec = ev.run_inference(tmp_video, tmp_path / "p.json", live=True, retry_failed=False,
                           budget={"calls": 0, "max_calls": 10})
    assert rec["status"] == "ok" and rec["response_id"] == "resp_x1"
    assert (rec["tokens_in"], rec["tokens_out"]) == (1234, 56) and rec["annotations"][0]["action"] == "grasp the brick"
    assert rec["model"] == "muse-spark-1.3-contributor" and isinstance(rec["qc_flags"], list)
    from robotrace.annotation_engine import OBSERVATION_PROMPT_VERSION
    assert rec["prompt_version"] == OBSERVATION_PROMPT_VERSION


# ------------------------------------------------------------------------------- MP4 / IVF helpers
def test_parse_moov_on_real_mp4(tiny_video):
    buf = tiny_video.read_bytes()
    moov = next((buf[o + 8:o + s] for typ, o, s in ev.top_level_boxes(buf) if typ == b"moov"))
    table = ev.parse_moov(moov)
    assert len(table["times"]) == len(table["sizes"]) == len(table["offsets"]) == 30
    assert table["times"][10] == pytest.approx(1.0, abs=1e-6) and 0 in table["keyframes"]
    idx, lo, hi = ev.episode_span(table, 0.0, 1.0)
    assert idx == list(range(10)) and lo < hi
    total = sum(table["sizes"][i] for i in idx)
    assert hi - lo >= total


def test_write_ivf_header(tmp_path):
    p = tmp_path / "x.ivf"
    ev.write_ivf(p, [b"ab", b"cde"], 640, 480, 30)
    b = p.read_bytes()
    assert b[:4] == b"DKIF" and b[8:12] == b"AV01" and len(b) == 32 + 12 + 2 + 12 + 3


def test_parse_episodes():
    assert ev.parse_episodes("0-2,5") == [0, 1, 2, 5]


@pytest.fixture
def tmp_video(tiny_video):
    return tiny_video
