"""Gemini adapter: request building, response parsing, conversion and failure handling.

The HTTP layer is faked. These tests check that RoboTrace handles Gemini's response FORMAT and ERRORS
correctly; they say nothing about whether Gemini's annotations of real robot videos are accurate.
"""
import json

import pytest
import requests

from robotrace import annotation_engine as ae
from robotrace import pipeline
from robotrace.config import Settings
from robotrace.pipeline import process_video
from robotrace.video_processor import probe

KEY = "AIza-test-key-not-real"


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setenv("ROBOTRACE_GEMINI_MODEL", "gemini-test-model")
    return Settings()


def answer(**over):
    data = {"scene_summary": "A robot arm pushes a red square on a grey table.", "robot_visible": True,
            "actions": [{"first_frame_t": 1.05, "last_frame_t": 1.95, "action": " push the red square ",
                         "objects": ["red square", " "], "visual_evidence": "square moves right with the arm",
                         "evidence_frame_times": [1.05, 1.95], "confidence": 0.7}]}
    data.update(over)
    return data


def api_body(data, finish="STOP", text=None, **extra):
    parts = [{"text": "thinking...", "thought": True},
             {"text": text if text is not None else json.dumps(data)}]
    return {"candidates": [{"content": {"parts": parts}, "finishReason": finish}],
            "modelVersion": "gemini-test-model-001", "responseId": "resp-123",
            "usageMetadata": {"promptTokenCount": 900, "candidatesTokenCount": 120}, **extra}


class FakeResp:
    def __init__(self, status, body):
        self.status_code = status
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        return json.loads(self.text)


def fake_post(monkeypatch, status=200, body=None, exc=None):
    seen = {}

    def post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, body=json, headers=headers, timeout=timeout)
        if exc:
            raise exc
        return FakeResp(status, body)

    monkeypatch.setattr(requests, "post", post)
    return seen


# ------------------------------------------------------------------------------------------- parsing
def test_parse_ignores_thought_parts_and_reads_metadata():
    data, meta = ae.parse_gemini_response(api_body(answer()))
    assert data["robot_visible"] is True
    assert meta == {"model_version": "gemini-test-model-001", "response_id": "resp-123",
                    "prompt_tokens": 900, "output_tokens": 120}


def test_parse_tolerates_markdown_fence():
    data, _ = ae.parse_gemini_response(api_body(None, text="```json\n" + json.dumps(answer()) + "\n```"))
    assert data["actions"][0]["confidence"] == 0.7


@pytest.mark.parametrize("body, match", [
    ({"promptFeedback": {"blockReason": "SAFETY"}}, "blocked"),
    ({"candidates": []}, "no candidates"),
    (api_body(answer(), finish="MAX_TOKENS"), "truncated"),
    (api_body(answer(), finish="SAFETY"), "finishReason=SAFETY"),
    (api_body(None, text="   "), "empty"),
    (api_body(None, text="{not json"), "malformed"),
    (api_body(None, text="[1, 2]"), "not an object"),
])
def test_parse_failures_raise(body, match):
    with pytest.raises(ae.AdapterUnavailable, match=match):
        ae.parse_gemini_response(body)


# -------------------------------------------------------------------------------------- HTTP / errors
def test_key_sent_in_header_not_url(keyed, monkeypatch):
    seen = fake_post(monkeypatch, body=api_body(answer()))
    ae.gemini_json_call(keyed, [{"text": "hi"}], {"type": "OBJECT"})
    assert seen["headers"] == {"x-goog-api-key": KEY} and KEY not in seen["url"]
    assert "models/gemini-test-model:generateContent" in seen["url"]
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"


@pytest.mark.parametrize("status, err, match", [
    (400, {"error": {"code": 400, "message": f"API key not valid: {KEY}", "status": "INVALID_ARGUMENT",
                     "details": [{"reason": "API_KEY_INVALID"}]}}, "rejected the API key"),
    (403, {"error": {"message": "denied", "status": "PERMISSION_DENIED"}}, "denied access"),
    (404, {"error": {"message": "not found", "status": "NOT_FOUND"}}, "gemini-test-model' was not found"),
    (429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}}, "quota or rate limit"),
    (503, "<html>unavailable</html>", "service error"),
])
def test_http_errors_are_reported_and_key_is_redacted(keyed, monkeypatch, status, err, match):
    fake_post(monkeypatch, status=status, body=err)
    with pytest.raises(ae.AdapterUnavailable, match=match) as ei:
        ae.gemini_json_call(keyed, [{"text": "hi"}], {"type": "OBJECT"})
    assert KEY not in str(ei.value) and f"HTTP {status}" in str(ei.value)


@pytest.mark.parametrize("exc, match", [(requests.Timeout(), "timed out"),
                                        (requests.ConnectionError(), "Could not reach")])
def test_network_failures(keyed, monkeypatch, exc, match):
    fake_post(monkeypatch, exc=exc)
    with pytest.raises(ae.AdapterUnavailable, match=match):
        ae.gemini_json_call(keyed, [{"text": "hi"}], {"type": "OBJECT"})


def test_missing_key_is_unavailable_and_makes_no_request(monkeypatch, tiny_video, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    seen = fake_post(monkeypatch, body=api_body(answer()))
    adapter = ae.GeminiFrameAdapter(Settings())
    assert adapter.availability()[0] is False
    with pytest.raises(ae.AdapterUnavailable, match="GEMINI_API_KEY"):
        adapter.annotate(tiny_video, probe(tiny_video), tmp_path)
    assert not seen


# ---------------------------------------------------------------------------------------- conversion
def test_conversion_marks_approximate_boundaries_and_drops_invented_frames():
    out = ae._GeminiOutput.model_validate(answer(actions=[
        {"first_frame_t": 0.5, "last_frame_t": 1.5, "action": "reach", "objects": [], "visual_evidence": "",
         "evidence_frame_times": [0.5, 7.77], "confidence": 0.5},
        {"first_frame_t": 2.5, "last_frame_t": 1.5, "action": "reversed", "objects": [], "visual_evidence": "",
         "evidence_frame_times": [], "confidence": 0.5}]))
    anns, dropped = ae.gemini_actions_to_annotations(out, [0.5, 1.5, 2.5], duration=3.0, step=1.0)
    assert (anns[0].start, anns[0].end, anns[0].boundary_precision_s) == (0.0, 2.0, 0.5)
    assert anns[0].cited_frame_times == [0.5] and dropped == 1
    assert (anns[1].start, anns[1].end) == (2.5, 1.5)  # kept reversed so QC flags it


def test_adapter_end_to_end_with_qc(keyed, monkeypatch, tiny_video):
    seen = fake_post(monkeypatch, body=api_body(answer()))
    qc_calls, real_qc = [], pipeline.run_qc
    monkeypatch.setattr(pipeline, "run_qc", lambda anns, *a, **k: qc_calls.append([x.id for x in anns])
                        or real_qc(anns, *a, **k))
    meta = probe(tiny_video)
    sess = process_video(tiny_video, meta, "gemini", settings=keyed)
    parts = seen["body"]["contents"][0]["parts"]
    images = [p for p in parts if "inlineData" in p]
    labels = [p["text"] for p in parts if p.get("text", "").startswith("t = ")]
    assert len(images) == len(labels) >= 4  # every frame is sent with its timestamp
    p = sess.processing
    assert p.adapter == "gemini" and p.is_live_inference and not p.is_precomputed_sample
    assert "resp-123" in p.adapter_detail and p.model_summary.startswith("A robot arm")
    a = sess.annotations[0]
    assert a.action == "push the red square" and a.objects == ["red square"]
    assert a.source == "gemini" and a.confidence_kind == "model_self_reported"
    assert a.boundary_precision_s is not None and a.evidence  # evidence frames attached as for other engines
    assert qc_calls == [[a.id]]  # deterministic QC ran on exactly the model's annotations
    assert any("approximate" in w for w in p.warnings)


def test_no_robot_visible_is_reported_not_hidden(keyed, monkeypatch, tiny_video, tmp_path):
    fake_post(monkeypatch, body=api_body(answer(robot_visible=False, actions=[])))
    res = ae.GeminiFrameAdapter(keyed).annotate(tiny_video, probe(tiny_video), tmp_path)
    assert res.annotations == [] and any("no robot is visible" in w for w in res.warnings)


def test_schema_violation_fails_instead_of_guessing(keyed, monkeypatch, tiny_video, tmp_path):
    fake_post(monkeypatch, body=api_body({"robot_visible": True, "actions": [{"action": "grab"}]}))
    with pytest.raises(ae.AdapterUnavailable, match="schema validation"):
        ae.GeminiFrameAdapter(keyed).annotate(tiny_video, probe(tiny_video), tmp_path)


def test_failed_engine_does_not_fall_back(keyed, monkeypatch, tiny_video):
    fake_post(monkeypatch, status=429, body={"error": {"message": "quota"}})
    with pytest.raises(ae.AdapterUnavailable):
        process_video(tiny_video, probe(tiny_video), "gemini", settings=keyed)
