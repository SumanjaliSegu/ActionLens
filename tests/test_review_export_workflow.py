"""Review -> export workflow: reviewer edits (incl. objects) reach every export; unresolved flags stay visible.

Covers the service layer and, via Streamlit's AppTest, the real Review-page edit form wiring.
"""
import csv
import io
import json
from pathlib import Path

import pytest
from conftest import ann
from streamlit.testing.v1 import AppTest

from robotrace import export_service as ex
from robotrace import review_service as rs
from robotrace.quality_engine import run_qc
from robotrace.schemas import ProcessingInfo, ReviewSession

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def session(meta):
    anns = [ann(0, 5, "grasp the red brick", id="a"), ann(4, 8, "place the brick", id="b"),
            ann(9, 8, "x", id="c")]
    anns[0].objects = ["red brick"]
    flags = run_qc(anns, meta.duration_s)
    return ReviewSession(video=meta, processing=ProcessingInfo(adapter="muse", is_live_inference=True),
                         baseline=[a.model_copy(deep=True) for a in anns],
                         annotations=[a.model_copy(deep=True) for a in anns],
                         initial_flags=[f.model_copy(deep=True) for f in flags], flags=flags)


def test_edits_persist_to_all_exports_and_open_flags_stay_visible(session):
    rs.edit_annotation(session, "a", start=0.5, end=3.9, action="grasp the pink brick", objects=["pink brick", " "])
    session.notes["a"] = "colour corrected"
    open_before = [f for f in session.flags if not f.resolved]
    assert len(open_before) >= 2
    rs.resolve_flag(session, open_before[0].id, note="checked")
    still_open = [f for f in session.flags if not f.resolved]
    assert still_open

    body = json.loads(ex.annotations_json(session))
    a = next(x for x in body["annotations"] if x["id"] == "a")
    assert (a["start"], a["end"], a["action"], a["objects"]) == (0.5, 3.9, "grasp the pink brick", ["pink brick"])
    assert a["source"].endswith("+edited") and a["reviewer_note"] == "colour corrected"

    rows = {r["id"]: r for r in csv.DictReader(io.StringIO(ex.annotations_csv(session)))}
    assert rows["a"]["objects"] == "pink brick" and rows["a"]["action"] == "grasp the pink brick"
    for x in session.annotations:
        n_open = sum(1 for f in session.flags if f.annotation_id == x.id and not f.resolved)
        assert int(rows[x.id]["open_flags"]) == n_open

    pkg = json.loads(ex.review_package_json(session))
    base = next(x for x in pkg["baseline_annotations"] if x["id"] == "a")
    assert (base["start"], base["action"], base["objects"]) == (0.0, "grasp the red brick", ["red brick"])
    current = {f["id"]: f for f in pkg["current_flags"]}
    assert all(current[f.id]["resolved"] is False for f in still_open)
    assert current[open_before[0].id]["resolved"] is True
    edit = next(e for e in pkg["review_events"] if e["kind"] == "edit")
    assert edit["payload"]["objects"] == [["red brick"], ["pink brick"]]

    md = ex.qc_report_markdown(session)
    assert any(f.code in md for f in still_open)


def test_objects_only_edit_is_not_dropped(session):
    rs.edit_annotation(session, "a", objects=["pink or red brick"])
    assert session.get("a").objects == ["pink or red brick"]
    assert session.events[-1].kind == "edit" and "objects" in session.events[-1].payload


def test_too_many_objects_rejected(session):
    with pytest.raises(rs.ReviewError):
        rs.edit_annotation(session, "a", objects=[f"o{i}" for i in range(11)])


@pytest.mark.skipif(not (ROOT / "data/sample/robotrace_sample.mp4").exists(), reason="no sample")
def test_review_form_edit_reaches_export_via_real_ui(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTRACE_WORK_DIR", str(tmp_path))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    next(b for b in at.button if b.label.startswith("▶ Try the sample")).click().run()
    sel = at.session_state["sel"]
    next(t for t in at.text_input if t.label == "Action description").set_value("grasp the pink brick")
    next(t for t in at.text_input if t.label.startswith("Objects")).set_value("pink brick, transparent box")
    next(b for b in at.button if b.label == "Save changes").click().run()
    assert not at.exception
    sess = at.session_state["review"]
    edited = sess.get(sel)
    assert edited.action == "grasp the pink brick" and edited.objects == ["pink brick", "transparent box"]
    exported = json.loads(ex.annotations_json(sess))
    assert next(x for x in exported["annotations"] if x["id"] == sel)["objects"] == ["pink brick", "transparent box"]
    open_codes = {f.code for f in sess.flags if not f.resolved}
    assert open_codes, "sample should still have unresolved flags"
    page_text = " ".join(m.value for m in at.markdown)
    assert any(code in page_text for code in open_codes)   # unresolved flags remain visible on the Review page
    at.sidebar.radio[0].set_value("Export").run()
    assert not at.exception


@pytest.mark.skipif(not (ROOT / "data/sample/robotrace_sample.mp4").exists(), reason="no sample")
def test_model_text_cannot_inject_markdown_links_or_images(tmp_path, monkeypatch):
    """Untrusted model output (scene summary, warnings, actions) must not render as live Markdown/HTML, which
    could make the reviewer's browser load external URLs."""
    monkeypatch.setenv("ROBOTRACE_WORK_DIR", str(tmp_path))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    next(b for b in at.button if b.label.startswith("▶ Try the sample")).click().run()
    evil = "![x](https://evil.example/p.png) [click](https://evil.example) <img src=x> *bold* $x$"
    sess = at.session_state["review"]
    sess.processing.model_summary = evil
    sess.processing.warnings.append(evil)
    sess.get(at.session_state["sel"]).action = evil
    at.run()
    assert not at.exception
    rendered = [e.value for e in list(at.markdown) + list(at.caption) + list(at.warning)]
    assert any("evil.example" in v for v in rendered)                  # the text is still shown to the reviewer
    for v in rendered:
        if "evil.example" in v:
            assert "](" not in v and "<img" not in v and "*bold*" not in v and "$x$" not in v


@pytest.mark.skipif(not (ROOT / "data/sample/robotrace_sample.mp4").exists(), reason="no sample")
def test_zero_annotations_with_no_actor_visible_gets_a_specific_reason(tmp_path, monkeypatch):
    """A vision engine correctly finding no robot AND no human in any frame (e.g. an empty/idle scene)
    produces zero annotations and a 'no robot or human actor visible' warning (see
    annotation_engine._annotate_frame_windows); the Analyze page should explain that specific reason, not
    just say 'no annotations' as if it were an unexplained failure. Covers the real user report where the
    engine was working correctly on an out-of-domain (non-robot) video, not a bug in parsing/storage —
    generalised so the message doesn't itself assume every video should contain a robot."""
    import robotrace.pipeline as pl
    monkeypatch.setenv("ROBOTRACE_WORK_DIR", str(tmp_path))
    real = pl.process_video

    def fake(*a, **k):
        sess = real(*a, **k)
        sess.annotations = []
        sess.processing.warnings = sess.processing.warnings + [
            "OpenCode Go (muse-spark-1.3-contributor) reported no robot or human actor visible in the "
            "sampled frames, so no actions were found. This is a valid result for a video with no visible "
            "manipulation, not a processing error."]
        return sess
    monkeypatch.setattr(pl, "process_video", fake)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Analyze").run()
    next(r for r in at.radio if r.label == "Source").set_value("Bundled synthetic sample").run()
    next(b for b in at.button if b.label == "Run analysis").click().run()
    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert any("no robot or human performing a visible" in w and "not an error" in w for w in warnings)
    assert not any(w == "The engine returned no annotations. You can still add annotations manually in Review."
                  for w in warnings)


@pytest.mark.skipif(not (ROOT / "data/sample/robotrace_sample.mp4").exists(), reason="no sample")
def test_zero_annotations_without_a_specific_reason_keeps_the_generic_message(tmp_path, monkeypatch):
    """Zero annotations for any OTHER reason (not 'no actor visible') still gets the generic message, not
    a misleading actor-specific explanation."""
    import robotrace.pipeline as pl
    monkeypatch.setenv("ROBOTRACE_WORK_DIR", str(tmp_path))
    real = pl.process_video

    def fake(*a, **k):
        sess = real(*a, **k)
        sess.annotations = []
        return sess
    monkeypatch.setattr(pl, "process_video", fake)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Analyze").run()
    next(r for r in at.radio if r.label == "Source").set_value("Bundled synthetic sample").run()
    next(b for b in at.button if b.label == "Run analysis").click().run()
    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert "The engine returned no annotations. You can still add annotations manually in Review." in warnings
    assert not any("no robot or human performing a visible" in w for w in warnings)


@pytest.mark.skipif(not (ROOT / "data/sample/robotrace_sample.mp4").exists(), reason="no sample")
def test_zero_annotations_with_actor_visible_but_no_action_keeps_the_generic_message(tmp_path, monkeypatch):
    """An actor IS visible (e.g. human_visible=True) but the engine still found nothing to annotate (idle
    footage): this is a different state from 'no actor at all' and must not claim no actor was seen."""
    import robotrace.pipeline as pl
    monkeypatch.setenv("ROBOTRACE_WORK_DIR", str(tmp_path))
    real = pl.process_video

    def fake(*a, **k):
        sess = real(*a, **k)
        sess.annotations = []  # no "no robot or human actor visible" warning added: an actor WAS seen
        return sess
    monkeypatch.setattr(pl, "process_video", fake)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Analyze").run()
    next(r for r in at.radio if r.label == "Source").set_value("Bundled synthetic sample").run()
    next(b for b in at.button if b.label == "Run analysis").click().run()
    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert "The engine returned no annotations. You can still add annotations manually in Review." in warnings
    assert not any("no robot or human" in w for w in warnings)


@pytest.mark.skipif(not (ROOT / "data/sample/robotrace_sample.mp4").exists(), reason="no sample")
def test_zero_annotation_session_does_not_crash_dashboard_or_export(tmp_path, monkeypatch):
    """Downstream Review/Dashboard/Export compatibility: a session with zero annotations (and the one QC
    gap flag that implies) must not crash any page, and the flag must render as a flag, not an error."""
    import robotrace.pipeline as pl
    monkeypatch.setenv("ROBOTRACE_WORK_DIR", str(tmp_path))
    real = pl.process_video

    def fake(*a, **k):
        sess = real(*a, **k)
        sess.annotations = []
        from robotrace.quality_engine import run_qc
        sess.flags = run_qc(sess.annotations, sess.video.duration_s, None, {})
        return sess
    monkeypatch.setattr(pl, "process_video", fake)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Analyze").run()
    next(r for r in at.radio if r.label == "Source").set_value("Bundled synthetic sample").run()
    next(b for b in at.button if b.label == "Run analysis").click().run()
    assert not at.exception
    for page in ("Review", "Dashboard", "Export"):
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page
    # the gap flag must appear as a flag (e.g. in the review queue's "unassigned findings"), never as
    # st.error, which would look like a processing failure rather than a QC finding
    assert not list(at.error)


@pytest.mark.skipif(not (ROOT / "data/sample/robotrace_sample.mp4").exists(), reason="no sample")
def test_analyze_page_escapes_model_text_in_warnings_and_engine_detail(tmp_path, monkeypatch):
    """Warnings, engine detail and scene summary are shown on the Analyze page right after a run."""
    import robotrace.pipeline as pl
    monkeypatch.setenv("ROBOTRACE_WORK_DIR", str(tmp_path))
    evil = "![x](https://evil.example/p.png) [click](https://evil.example) <img src=x>"
    real = pl.process_video

    def fake(*a, **k):
        sess = real(*a, **k)
        sess.processing.warnings.append(f"Object names differ between windows: {evil}")
        sess.processing.adapter_detail = f"served as {evil}"
        sess.processing.model_summary = evil
        return sess
    monkeypatch.setattr(pl, "process_video", fake)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Analyze").run()
    next(r for r in at.radio if r.label == "Source").set_value("Bundled synthetic sample").run()
    next(b for b in at.button if b.label == "Run analysis").click().run()
    assert not at.exception
    shown = {"warning": [w.value for w in at.warning if "evil.example" in w.value],
             "caption": [c.value for c in at.caption if "evil.example" in c.value],
             "markdown": [m.value for m in at.markdown if "evil.example" in m.value]}
    assert shown["warning"] and shown["caption"] and shown["markdown"]    # all three call sites rendered
    for v in sum(shown.values(), []):
        assert "](" not in v and "<img" not in v
