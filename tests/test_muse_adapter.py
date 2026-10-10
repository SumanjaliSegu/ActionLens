"""Muse Spark via OpenCode Go: request building, response parsing, conversion and failure handling.

The HTTP layer is faked. These tests check that RoboTrace handles the Responses-API FORMAT and ERRORS
correctly; they say nothing about whether Muse Spark's annotations of real robot videos are accurate.
"""
import json

import pytest
import requests

from robotrace import annotation_engine as ae
from robotrace import pipeline
from robotrace.config import Settings
from robotrace.pipeline import process_video
from robotrace.video_processor import probe

KEY = "oc-go-test-key-not-real"
GEMINI_KEY = "AIza-should-never-be-used"


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", KEY)
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)  # present, to prove Gemini is not called
    monkeypatch.delenv("ROBOTRACE_MUSE_MODEL", raising=False)
    monkeypatch.delenv("ROBOTRACE_OPENCODE_GO_URL", raising=False)
    return Settings()


def answer(**over):
    data = {"scene_summary": "A robot arm pushes a red square on a grey table.", "robot_visible": True,
            "actions": [{"first_frame_t": 1.05, "last_frame_t": 1.95, "action": "push the red square",
                         "objects": ["red square"], "visual_evidence": "square moves right with the gripper",
                         "evidence_frame_times": [1.05, 1.95], "confidence": 0.6}]}
    data.update(over)
    return data


def api_body(data=None, text=None, status="completed", **extra):
    msg = {"type": "message", "role": "assistant",
           "content": [{"type": "output_text", "text": text if text is not None else json.dumps(data)}]}
    body = {"id": "resp_muse_123", "object": "response", "model": "muse-spark-1.3-contributor",
            "status": status, "output": [{"type": "reasoning", "summary": []}, msg],
            "usage": {"input_tokens": 1500, "output_tokens": 300}}
    body.update(extra)
    return body


class FakeResp:
    def __init__(self, status, body):
        self.status_code = status
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        return json.loads(self.text)


def fake_post(monkeypatch, status=200, body=None, exc=None):
    calls = []

    def post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "body": json, "headers": headers, "timeout": timeout})
        if exc:
            raise exc
        return FakeResp(status, body)

    monkeypatch.setattr(requests, "post", post)
    return calls


# ------------------------------------------------------------------------------------------- parsing
def test_parse_reads_text_and_metadata_and_skips_reasoning():
    data, meta = ae.parse_opencode_response(api_body(answer()))
    assert data["actions"][0]["action"] == "push the red square"
    assert meta == {"model_version": "muse-spark-1.3-contributor", "response_id": "resp_muse_123",
                    "prompt_tokens": 1500, "output_tokens": 300}


@pytest.mark.parametrize("text", ["```json\n" + json.dumps(answer()) + "\n```",
                                  "Here are the observations:\n" + json.dumps(answer())])
def test_parse_tolerates_fences_and_surrounding_prose(text):
    data, _ = ae.parse_opencode_response(api_body(text=text))
    assert data["robot_visible"] is True


@pytest.mark.parametrize("body, match", [
    (api_body(answer(), error={"message": "upstream failed"}), "upstream failed"),
    (api_body(answer(), status="incomplete", incomplete_details={"reason": "max_output_tokens"}), "truncated"),
    (api_body(answer(), status="incomplete", incomplete_details={"reason": "content_filter"}), "content_filter"),
    (api_body(answer(), status="failed"), "status=failed"),
    ({"status": "completed", "output": [{"type": "message", "content": [
        {"type": "refusal", "refusal": "I can't help with that."}]}]}, "declined"),
    (api_body(text="  "), "empty"),
    (api_body(text="no json here"), "malformed"),
    (api_body(text="[1, 2]"), "not an object"),
])
def test_parse_failures_raise(body, match):
    with pytest.raises(ae.AdapterUnavailable, match=match):
        ae.parse_opencode_response(body)


# -------------------------------------------------------------------------------------- HTTP / errors
def test_request_format_auth_and_headers(keyed, monkeypatch):
    calls = fake_post(monkeypatch, body=api_body(answer()))
    ae.opencode_go_call(keyed, [{"type": "input_text", "text": "hi"}])
    c = calls[0]
    assert c["url"] == "https://opencode.ai/zen/go/v1/responses"
    assert c["headers"]["Authorization"] == f"Bearer {KEY}" and KEY not in c["url"]
    assert c["headers"]["User-Agent"].startswith("RoboTrace/") and c["headers"]["x-opencode-session"]
    assert c["body"]["model"] == "muse-spark-1.3-contributor"
    assert c["body"]["input"][0]["role"] == "user"


def test_session_header_is_stable_across_calls(keyed, monkeypatch):
    calls = fake_post(monkeypatch, body=api_body(answer()))
    ae.opencode_go_call(keyed, [{"type": "input_text", "text": "a"}])
    ae.opencode_go_call(keyed, [{"type": "input_text", "text": "b"}])
    assert calls[0]["headers"]["x-opencode-session"] == calls[1]["headers"]["x-opencode-session"]


@pytest.mark.parametrize("status, err, match", [
    (401, {"type": "error", "error": {"type": "AuthError", "message": f"Invalid API key {KEY}"}},
     "rejected the API key"),
    (403, {"error": {"type": "Forbidden", "message": "region"}}, "refused the request"),
    (404, {"error": {"message": "model not found"}}, "muse-spark-1.3-contributor' was not found"),
    (429, {"error": {"message": "limit"}}, "usage or rate limit"),
    (400, {"error": {"message": "image input not supported"}}, "image input not supported"),
    (502, "<html>bad gateway</html>", "service error"),
])
def test_http_errors_are_reported_and_key_is_redacted(keyed, monkeypatch, status, err, match):
    fake_post(monkeypatch, status=status, body=err)
    with pytest.raises(ae.AdapterUnavailable, match=match) as ei:
        ae.opencode_go_call(keyed, [{"type": "input_text", "text": "hi"}])
    assert KEY not in str(ei.value) and f"HTTP {status}" in str(ei.value)


@pytest.mark.parametrize("exc, match", [(requests.Timeout(), "timed out"),
                                        (requests.ConnectionError(), "Could not reach")])
def test_network_failures(keyed, monkeypatch, exc, match):
    fake_post(monkeypatch, exc=exc)
    with pytest.raises(ae.AdapterUnavailable, match=match):
        ae.opencode_go_call(keyed, [{"type": "input_text", "text": "hi"}])


def test_missing_key_is_unavailable_and_makes_no_request(monkeypatch, tiny_video, tmp_path):
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    calls = fake_post(monkeypatch, body=api_body(answer()))
    adapter = ae.MuseSparkAdapter(Settings())
    assert adapter.availability()[0] is False
    with pytest.raises(ae.AdapterUnavailable, match="OPENCODE_GO_API_KEY"):
        adapter.annotate(tiny_video, probe(tiny_video), tmp_path)
    assert calls == []


# ----------------------------------------------------------------------------------------- end to end
def test_end_to_end_sends_timestamped_frames_runs_qc_and_never_calls_gemini(keyed, monkeypatch, tiny_video):
    calls = fake_post(monkeypatch, body=api_body(answer()))
    qc_calls, real_qc = [], pipeline.run_qc
    monkeypatch.setattr(pipeline, "run_qc", lambda anns, *a, **k: qc_calls.append([x.id for x in anns])
                        or real_qc(anns, *a, **k))
    sess = process_video(tiny_video, probe(tiny_video), "muse", settings=keyed)

    assert len(calls) == 1 and "opencode.ai/zen/go/v1/responses" in calls[0]["url"]
    assert not any("googleapis" in c["url"] for c in calls)
    assert all(GEMINI_KEY not in json.dumps(c["headers"]) for c in calls)
    content = calls[0]["body"]["input"][0]["content"]
    images = [c for c in content if c["type"] == "input_image"]
    labels = [c["text"] for c in content if c["type"] == "input_text" and c["text"].startswith("t = ")]
    assert len(images) == len(labels) >= 4
    assert all(i["image_url"].startswith("data:image/jpeg;base64,") for i in images)

    p = sess.processing
    assert (p.adapter, p.provider, p.model) == ("muse", "OpenCode Go", "muse-spark-1.3-contributor")
    assert p.prompt_version == ae.OBSERVATION_PROMPT_VERSION == "obs-v3"
    prompt = " ".join(content[0]["text"].split())
    assert "never add a step that no frame shows" in prompt          # distinct steps, but only visible ones
    assert "first frame where the arm has visibly moved" in prompt   # exclude initial still frames
    assert "if unsure, say so" in prompt                             # hedge uncertain colours
    assert p.is_live_inference and not p.is_precomputed_sample and "resp_muse_123" in p.adapter_detail
    assert any("train future Meta models" in w for w in p.warnings)
    a = sess.annotations[0]
    assert a.source == "muse" and a.objects == ["red square"] and a.boundary_precision_s is not None
    assert qc_calls == [[a.id]]  # deterministic QC ran on exactly the model's annotations


def test_schema_violation_fails_instead_of_guessing(keyed, monkeypatch, tiny_video, tmp_path):
    fake_post(monkeypatch, body=api_body({"robot_visible": True, "actions": [{"action": "grab"}]}))
    with pytest.raises(ae.AdapterUnavailable, match="schema validation"):
        ae.MuseSparkAdapter(keyed).annotate(tiny_video, probe(tiny_video), tmp_path)


def test_failed_muse_does_not_fall_back(keyed, monkeypatch, tiny_video):
    calls = fake_post(monkeypatch, status=401, body={"error": {"type": "AuthError", "message": "Invalid API key."}})
    with pytest.raises(ae.AdapterUnavailable, match="rejected the API key"):
        process_video(tiny_video, probe(tiny_video), "muse", settings=keyed)
    assert len(calls) == 1  # no retry against Gemini or any other provider


def test_model_override_from_env(monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", KEY)
    monkeypatch.setenv("ROBOTRACE_MUSE_MODEL", "muse-spark-1.2-contributor")
    ok, why = ae.MuseSparkAdapter(Settings()).availability()
    assert ok and "muse-spark-1.2-contributor" in why and KEY not in why
