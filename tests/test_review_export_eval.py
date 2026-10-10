import csv
import io
import json

import pytest
from conftest import ann

from robotrace import export_service as ex
from robotrace import review_service as rs
from robotrace.evaluation import Segment, evaluate_flags, evaluate_segments, iou, normalize_label
from robotrace.quality_engine import run_qc
from robotrace.schemas import SCHEMA_VERSION, Annotation, ProcessingInfo, ReviewSession, ReviewStatus


@pytest.fixture
def session(meta):
    anns = [ann(0, 5, "reach the cube", id="a"), ann(4, 8, "=HYPERLINK(evil)", id="b"), ann(9, 8, "x", id="c")]
    flags = run_qc(anns, meta.duration_s)
    return ReviewSession(video=meta, processing=ProcessingInfo(adapter="test", is_live_inference=False),
                         baseline=[a.model_copy(deep=True) for a in anns],
                         annotations=[a.model_copy(deep=True) for a in anns],
                         initial_flags=[f.model_copy(deep=True) for f in flags], flags=flags)


def test_annotation_schema_validation():
    a = Annotation(start=1, end=2, action="x")
    assert a.id.startswith("ann_") and a.confidence_kind == "none" and a.duration == 1
    with pytest.raises(Exception):
        Annotation(start="soon", end=2, action="x")


def test_status_change_logged(session):
    rs.set_status(session, "a", ReviewStatus.correct, "looks right")
    assert session.status["a"] == ReviewStatus.correct
    assert session.events[-1].kind == "status" and session.events[-1].payload["to"] == "correct"
    with pytest.raises(KeyError):
        rs.set_status(session, "nope", "correct")


def test_edit_validates_and_rechecks(session):
    assert any(f.code == "TS_REVERSED" for f in session.flags)
    rs.edit_annotation(session, "c", start=8.5, end=9.5)
    assert not any(f.code == "TS_REVERSED" for f in session.flags)
    assert session.baseline[2].start == 9  # baseline untouched
    assert session.get("c").source.endswith("+edited")
    for bad in ({"start": -1}, {"end": 0.0}, {"end": 99}, {"action": "  "}):
        with pytest.raises(rs.ReviewError):
            rs.edit_annotation(session, "a", **bad)


def test_edit_fixing_overlap_removes_flag(session):
    rs.edit_annotation(session, "a", end=4)
    assert not any(f.code == "TS_OVERLAP" for f in session.flags)
    # initial flags are frozen for before/after comparison
    assert any(f.code == "TS_OVERLAP" for f in session.initial_flags)


def test_add_and_delete(session):
    new = rs.add_annotation(session, 8.0, 9.0, "place the cube")
    assert session.status[new.id] == ReviewStatus.correct and new.source == "human"
    rs.delete_annotation(session, "b")
    assert "b" in session.deleted_ids and all(a.id != "b" for a in session.annotations)
    with pytest.raises(rs.ReviewError):
        rs.add_annotation(session, 5, 4, "bad")


def test_resolve_flag_and_summary(session):
    fid = session.flags[0].id
    rs.resolve_flag(session, fid, "false alarm")
    s = rs.summary(session)
    assert s["flags_current"]["open"] == s["flags_current"]["total"] - 1
    assert s["review_actions"]["resolve_flag"] == 1
    with pytest.raises(rs.ReviewError):
        rs.resolve_flag(session, "missing")


def test_exports_are_consistent(session):
    rs.set_status(session, "a", "correct")
    j = json.loads(ex.annotations_json(session))
    assert j["schema_version"] == SCHEMA_VERSION and j["time_unit"] == "seconds"
    assert j["video"]["sha256"] == session.video.sha256
    assert {a["id"]: a["review_status"] for a in j["annotations"]}["a"] == "correct"
    rows = list(csv.DictReader(io.StringIO(ex.annotations_csv(session))))
    assert len(rows) == 3 and rows[0]["start_s"] == "0.0"
    assert rows[1]["action"].startswith("'=")  # formula injection neutralised
    pkg = json.loads(ex.review_package_json(session))
    assert {"baseline_annotations", "reviewed_annotations", "initial_flags", "current_flags",
            "review_events", "summary"} <= set(pkg)
    report = ex.qc_report_markdown(session)
    assert "# RoboTrace Quality-Control Report" in report and "TS_REVERSED" in report


def test_export_strips_local_paths(session):
    from robotrace.schemas import EvidenceFrame
    session.annotations[0].evidence = [EvidenceFrame(t=1.0, frame_index=10, path="/srv/secret/f.jpg")]
    assert "/srv/secret" not in ex.annotations_json(session)
    assert "/srv/secret" not in ex.review_package_json(session)


def test_iou_and_label_normalisation():
    assert iou((0, 2), (1, 3)) == pytest.approx(1 / 3)
    assert iou((0, 1), (2, 3)) == 0
    assert normalize_label("Grasp the Red cube!") == normalize_label("grasp red cube")


def test_evaluate_segments():
    ref = [Segment(0, 2, "reach"), Segment(2, 4, "grasp"), Segment(4, 6, "lift")]
    pred = [ann(0, 2, "reach"), ann(2.2, 4, "grasp"), ann(7, 8, "wave"), ann(5, 4, "lift")]
    e = evaluate_segments(pred, ref)
    assert len(e.matches) == 2 and e.precision == 0.5 and e.recall == pytest.approx(0.667, abs=1e-3)
    assert e.label_accuracy == 1.0 and e.mean_start_error_s == pytest.approx(0.1)


def test_evaluate_flags():
    anns = [ann(0, 1, id="a"), ann(1, 2, id="b"), ann(3, 2, id="c")]
    flags = run_qc(anns, 10.0)
    out = evaluate_flags(flags, anns, {"c", "b"}, [Segment(5, 6, "missing")])
    assert out["true_positive"] == 1 and out["false_negative"] == 1 and out["flag_precision"] == 1.0
    assert out["missing_with_candidate_or_gap_flag"] == 1
