"""Claude adapter / verifier logic with the network call mocked (no API key or cost needed)."""
import pytest

from robotrace import annotation_engine as ae
from robotrace import quality_engine as qe
from robotrace.config import Settings
from robotrace.video_processor import probe


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    return Settings()


def test_claude_adapter_parses_and_labels_confidence(keyed, monkeypatch, tiny_video, tmp_path):
    seen = {}

    def fake_call(settings, content, schema, max_tokens=16000):
        seen["images"] = sum(1 for c in content if c["type"] == "image")
        return {"segments": [{"start": 1.0, "end": 2.0, "action": " push the red square ", "confidence": 0.8}]}

    monkeypatch.setattr(ae, "claude_json_call", fake_call)
    res = ae.ClaudeFrameAdapter(keyed).annotate(tiny_video, probe(tiny_video), tmp_path)
    a = res.annotations[0]
    assert (a.start, a.end, a.action) == (1.0, 2.0, "push the red square")
    assert a.confidence_kind == "model_self_reported" and res.is_live_inference
    assert seen["images"] >= 3 and any("not calibrated" in w for w in res.warnings)


def test_claude_adapter_rejects_schema_violations(keyed, monkeypatch, tiny_video, tmp_path):
    monkeypatch.setattr(ae, "claude_json_call", lambda *a, **k: {"segments": [{"start": "soon"}]})
    with pytest.raises(ae.AdapterUnavailable):
        ae.ClaudeFrameAdapter(keyed).annotate(tiny_video, probe(tiny_video), tmp_path)


def test_verifier_ignores_invented_ids(keyed, monkeypatch, tiny_video, tmp_path):
    meta = probe(tiny_video)
    anns = [ae.Annotation(start=1, end=2, action="push the square", id="a1")]
    ae.attach_evidence(anns, tiny_video, meta, tmp_path)
    monkeypatch.setattr(ae, "claude_json_call", lambda *a, **k: {"verdicts": [
        {"id": "a1", "verdict": "contradicted", "observed": "square slides", "explanation": "x"},
        {"id": "made_up", "verdict": "supported", "observed": "", "explanation": ""}]})
    out = qe.model_verify(anns, keyed)
    assert set(out) == {"a1"} and out["a1"]["verdict"] == "contradicted"
    flags = qe.model_flags(anns, out)
    assert flags[0].code == "MODEL_CONTRADICTED"


def test_refusal_becomes_unavailable(keyed, monkeypatch):
    class Resp:
        stop_reason = "refusal"
        content = []

    class FakeClient:
        def __init__(self, *a, **k):
            self.beta = type("B", (), {"messages": type("M", (), {"create": staticmethod(lambda **k: Resp())})()})()

    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    with pytest.raises(ae.AdapterUnavailable, match="declined"):
        ae.claude_json_call(keyed, [{"type": "text", "text": "hi"}], {"type": "object"})
