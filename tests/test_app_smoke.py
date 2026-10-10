"""Headless UI smoke test: drives the real Streamlit script through the main workflow."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
pytestmark = pytest.mark.skipif(not (ROOT / "data/sample/robotrace_sample.mp4").exists(), reason="no sample")


def _button(at, startswith):
    return next(b for b in at.button if b.label.startswith(startswith))


def test_workflow_sample_review_dashboard_export(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTRACE_WORK_DIR", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("RoboTrace" in m.value for m in at.markdown)

    _button(at, "▶ Try the sample").click().run()
    assert not at.exception
    assert at.session_state["page"] == "Review"
    sess = at.session_state["review"]
    assert sess.processing.is_precomputed_sample and len(sess.annotations) == 11

    # top of the queue is a HIGH-severity item; mark it incorrect via the real button
    sel = at.session_state["sel"]
    _button(at, "✗ Incorrect").click().run()
    assert not at.exception
    assert at.session_state["review"].status[sel].value == "incorrect"

    # select another annotation from the queue and resolve its first flag
    target = next(a for a in sess.annotations if a.action.startswith("grasp the red"))
    next(b for b in at.button if b.key == f"q_{target.id}").click().run()
    assert at.session_state["sel"] == target.id
    flag = next(f for f in at.session_state["review"].flags if f.annotation_id == target.id)
    next(b for b in at.button if b.key == f"res_{flag.id}").click().run()
    assert next(f for f in at.session_state["review"].flags if f.id == flag.id).resolved

    for page in ("Dashboard", "Export", "Analyze", "Overview"):
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page
    at.sidebar.radio[0].set_value("Dashboard").run()
    assert any("Evaluation against reference" in m.value for m in at.markdown)


def test_pages_without_session_show_empty_state(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTRACE_WORK_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    for page in ("Review", "Dashboard", "Export", "Analyze"):
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page
