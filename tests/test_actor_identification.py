"""Actor-generalised annotation (obs-v4): robot, human/egocentric, mixed, ambiguous and no-actor scenes.

Covers the design limitation found via a real user report: the prompt previously framed every video as
"a robot manipulation video" and asked only for "robot actions", so a vision model correctly following
those instructions returned an empty actions list for any video where no robot was visible — even when a
human was clearly performing describable manipulation (hands assembling hardware on a workbench). The
model's own behavior was correct given that prompt; the prompt was the bug. These tests cover the fix:
describing actions by any visible actor (robot and/or human), never inventing or relabeling an actor's
identity, and distinguishing "no actor was visible" from "an actor was visible but did nothing notable"
from "the video could not be processed at all".

HTTP is faked throughout (same pattern as test_muse_adapter.py); no live API calls.
"""
import json

import pytest
import requests

from robotrace import annotation_engine as ae
from robotrace.config import Settings
from robotrace.pipeline import process_video
from robotrace.sampling import merge_window_annotations
from robotrace.schemas import Annotation
from robotrace.video_processor import VideoError, probe

KEY = "oc-go-test-key-not-real"


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", KEY)
    monkeypatch.delenv("ROBOTRACE_MUSE_MODEL", raising=False)
    monkeypatch.delenv("ROBOTRACE_OPENCODE_GO_URL", raising=False)
    monkeypatch.setenv("ROBOTRACE_SAMPLING", "legacy")  # one request; see test_muse_adapter.py for why
    return Settings()


class FakeResp:
    def __init__(self, status, body):
        self.status_code = status
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        return json.loads(self.text)


def fake_post(monkeypatch, body, status=200):
    calls = []

    def post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        return FakeResp(status, body)
    monkeypatch.setattr(requests, "post", post)
    return calls


def api_body(data):
    return {"id": "resp_x1", "object": "response", "model": "muse-spark-1.3-contributor", "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(data)}]}],
            "usage": {"input_tokens": 500, "output_tokens": 80}}


def act(first_t, last_t, actor, action, confidence=0.8, **over):
    d = {"first_frame_t": first_t, "last_frame_t": last_t, "actor": actor, "action": action,
         "objects": [], "visual_evidence": "", "evidence_frame_times": [first_t], "confidence": confidence}
    d.update(over)
    return d


# --------------------------------------------------------------------------------- robot (regression)
def test_robot_manipulation_regression(keyed, monkeypatch, tiny_video):
    """Existing robot-visible behavior is unchanged: annotations are produced and tagged actor='robot'."""
    fake_post(monkeypatch, api_body({
        "scene_summary": "A robot arm pushes a red square on a grey table.",
        "robot_visible": True, "human_visible": False,
        "actions": [act(1.05, 1.95, "robot", "push the red square")]}))
    sess = process_video(tiny_video, probe(tiny_video), "muse", settings=keyed)
    assert len(sess.annotations) == 1
    assert sess.annotations[0].actor == "robot"
    assert not any("no robot or human" in w for w in sess.processing.warnings)


# --------------------------------------------------------------------------------- egocentric / human
def test_egocentric_human_hands_manipulation_is_annotated_not_suppressed(keyed, monkeypatch, tiny_video):
    """The actual bug: a human-hands video with NO robot visible must still be annotated, tagged
    actor='human', not silently returned as zero annotations."""
    fake_post(monkeypatch, api_body({
        "scene_summary": "A person's hands pick up a small bracket and attach it to a frame on a workbench.",
        "robot_visible": False, "human_visible": True,
        "actions": [act(1.05, 1.95, "human", "grasp the bracket"),
                   act(2.0, 2.9, "human", "attach the bracket to the frame")]}))
    sess = process_video(tiny_video, probe(tiny_video), "muse", settings=keyed)
    assert len(sess.annotations) == 2
    assert all(a.actor == "human" for a in sess.annotations)
    assert not any("no robot or human" in w for w in sess.processing.warnings)
    assert not any("The engine returned no annotations" in w for w in sess.processing.warnings)


def test_human_only_activity_never_mislabeled_as_robot(keyed, monkeypatch, tiny_video):
    """A clearly-human actor must never be stored as actor='robot', even though the model is free to use
    its own wording ('a person's hand') -- canonical_actor must map it correctly."""
    fake_post(monkeypatch, api_body({
        "scene_summary": "Hands manipulate tools.", "robot_visible": False, "human_visible": True,
        "actions": [act(1.0, 2.0, "a person's hand", "pick up the screwdriver")]}))
    sess = process_video(tiny_video, probe(tiny_video), "muse", settings=keyed)
    assert sess.annotations[0].actor == "human"


# --------------------------------------------------------------------------------- no actor / ambiguous
def test_no_actor_visible_is_zero_annotations_with_specific_warning(keyed, monkeypatch, tiny_video):
    """Both robot_visible and human_visible false, actions=[]: a legitimate zero-annotation success with
    a specific, non-alarming reason -- not a technical failure."""
    fake_post(monkeypatch, api_body({"scene_summary": "An empty workbench, nothing moving.",
                                     "robot_visible": False, "human_visible": False, "actions": []}))
    sess = process_video(tiny_video, probe(tiny_video), "muse", settings=keyed)
    assert sess.annotations == []
    assert any("no robot or human actor visible" in w for w in sess.processing.warnings)
    assert any("not a processing error" in w for w in sess.processing.warnings)


def test_actor_visible_but_no_action_is_zero_annotations_without_the_no_actor_warning(keyed, monkeypatch, tiny_video):
    """An actor IS visible (idle human) but there's genuinely nothing to annotate: zero annotations, but
    the 'no robot or human actor visible' warning must NOT fire -- an actor was seen, just not acting."""
    fake_post(monkeypatch, api_body({"scene_summary": "A person stands at the bench, hands at their sides.",
                                     "robot_visible": False, "human_visible": True, "actions": []}))
    sess = process_video(tiny_video, probe(tiny_video), "muse", settings=keyed)
    assert sess.annotations == []
    assert not any("no robot or human actor visible" in w for w in sess.processing.warnings)


def test_ambiguous_actor_is_recorded_as_unclear_not_invented(keyed, monkeypatch, tiny_video):
    """When the frames don't make the actor clearly robot or human, the model is asked to say 'unclear'
    with lower confidence -- the action is still recorded (not invented from nothing, not silently
    dropped), and nothing asserts an identity the evidence doesn't support."""
    fake_post(monkeypatch, api_body({
        "scene_summary": "A manipulator (type unclear from this angle) moves an object.",
        "robot_visible": False, "human_visible": False,
        "actions": [act(1.0, 2.0, "unclear", "move the object", confidence=0.3)]}))
    sess = process_video(tiny_video, probe(tiny_video), "muse", settings=keyed)
    assert len(sess.annotations) == 1
    assert sess.annotations[0].actor == "unclear"
    assert sess.annotations[0].confidence == 0.3


@pytest.mark.parametrize("raw, expected", [
    ("robot", "robot"), ("the robot gripper", "robot"), ("robotic arm", "robot"), ("a manipulator arm", "robot"),
    ("human", "human"), ("a human hand", "human"), ("person's hands", "human"), ("the operator", "human"),
    ("unclear", "unclear"), ("ambiguous", "unclear"), ("uncertain actor", "unclear"),
    ("", ""), ("   ", ""),
    ("a dog", "unclear"),  # unrecognised phrasing: never silently defaults to robot or human
])
def test_canonical_actor_normalisation(raw, expected):
    assert ae.canonical_actor(raw) == expected


# --------------------------------------------------------------------------------- merge across windows
def A(s, e, action, actor="", conf=0.8):
    return Annotation(start=s, end=e, action=action, actor=actor, confidence=conf)


def test_cross_window_egocentric_action_merges_and_keeps_agreeing_actor():
    out, n = merge_window_annotations(
        [[A(8.0, 11.9, "carry the bracket to the bin", actor="human")],
         [A(11.2, 14.0, "carry the bracket to the bin", actor="human")]], tol=0.25)
    assert n == 1 and len(out) == 1
    assert out[0].actor == "human"


def test_disagreeing_actor_across_windows_resets_to_unspecified_not_a_guess():
    """If merged windows disagree on who performed the SAME action, don't assert either identity."""
    out, n = merge_window_annotations(
        [[A(8.0, 11.9, "move the part", actor="human")],
         [A(11.2, 14.0, "move the part", actor="robot")]], tol=0.25)
    assert n == 1 and len(out) == 1
    assert out[0].actor == ""


def test_missing_actor_in_one_window_does_not_overwrite_a_known_one():
    out, n = merge_window_annotations(
        [[A(8.0, 11.9, "move the part", actor="human")],
         [A(11.2, 14.0, "move the part", actor="")]], tol=0.25)
    assert n == 1 and out[0].actor == "human"


# --------------------------------------------------------------------------------- technical failure
def test_technical_decode_failure_is_distinct_from_a_legitimate_zero_annotation_result(tmp_path):
    """A corrupt/undecodable video raises VideoError (a technical failure) long before any annotation
    engine runs -- it must never be reported as a successful run with zero annotations."""
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video at all" * 100)
    with pytest.raises(VideoError):
        probe(bad)
