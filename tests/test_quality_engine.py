from conftest import ann, make_signal

from robotrace.quality_engine import (carry_resolutions, priority_score, rank_annotations, run_qc,
                                      signal_flags, timestamp_flags, uncovered_intervals, model_flags, snapshot)
from robotrace.schemas import Flag, FlagOrigin, Severity


def codes(flags, ann_id=None):
    return sorted(f.code for f in flags if ann_id is None or f.annotation_id == ann_id)


def test_clean_annotations_have_no_timestamp_flags():
    anns = [ann(0, 3), ann(3, 6), ann(6, 10)]
    assert timestamp_flags(anns, 10.0) == []


def test_reversed_negative_out_of_range_zero_length():
    a, b, c, d = ann(5, 4, id="rev"), ann(-1, 2, id="neg"), ann(8, 12, id="oor"), ann(3, 3, id="zero")
    flags = timestamp_flags([a, b, c, d], 10.0)
    assert "TS_REVERSED" in codes(flags, "rev")
    assert "TS_NEGATIVE" in codes(flags, "neg")
    assert "TS_OUT_OF_RANGE" in codes(flags, "oor")
    assert "TS_ZERO_LENGTH" in codes(flags, "zero")
    assert all(f.origin == FlagOrigin.rule for f in flags)


def test_overlap_flags_both_annotations_with_reference():
    a, b = ann(0, 5, id="a"), ann(4, 8, id="b")
    flags = [f for f in timestamp_flags([a, b], 10.0) if f.code == "TS_OVERLAP"]
    assert {f.annotation_id for f in flags} == {"a", "b"}
    assert flags[0].details["overlap_s"] == 1.0


def test_touching_intervals_do_not_overlap():
    assert not [f for f in timestamp_flags([ann(0, 5), ann(5, 9)], 10.0) if f.code == "TS_OVERLAP"]


def test_gap_is_reported_but_not_called_missing():
    flags = timestamp_flags([ann(0, 2), ann(6, 10)], 10.0)
    gaps = [f for f in flags if f.code == "TS_GAP"]
    assert len(gaps) == 1 and (gaps[0].start, gaps[0].end) == (2, 6)
    assert "not by itself evidence of a missing action" in gaps[0].reason
    assert not [f for f in flags if f.category == "missing_candidate"]


def test_long_segment_relative_to_median():
    anns = [ann(0, 1), ann(1, 2), ann(2, 3), ann(3, 9.5)]
    assert "TS_LONG_SEGMENT" in codes(timestamp_flags(anns, 10.0))


def test_uncovered_intervals_ignores_invalid_annotations():
    gaps = uncovered_intervals([ann(0, 2), ann(5, 4), ann(6, 10)], 10.0)
    assert gaps == [(2, 6)]


def test_low_motion_static_is_medium_subtle_is_low():
    sig = make_signal(active_spans=[(4, 6)], subtle_spans=[(7, 8)])
    static, subtle, moving = ann(0.5, 2, id="static"), ann(7, 8, "grasp the cube", id="subtle"), ann(4, 6, id="mv")
    flags = signal_flags([static, subtle, moving], sig, 10.0)
    low = {f.annotation_id: f.severity for f in flags if f.code == "EV_LOW_MOTION"}
    assert low == {"static": Severity.medium, "subtle": Severity.low}
    assert all(f.origin == FlagOrigin.signal for f in flags)


def test_idle_words_suppress_low_motion_flag():
    sig = make_signal(active_spans=[(4, 6)])
    flags = signal_flags([ann(0.5, 2, "robot idle, waiting")], sig, 10.0)
    assert "EV_LOW_MOTION" not in codes(flags)


def test_missing_candidate_only_for_uncovered_motion():
    sig = make_signal(active_spans=[(3, 5), (6, 10)])
    anns = [ann(3, 5, id="a")]  # motion at 7-9 is not annotated
    flags = signal_flags(anns, sig, 10.0)
    cands = [f for f in flags if f.code == "MISSING_CANDIDATE"]
    assert len(cands) == 1 and cands[0].start == 5 and cands[0].end == 10
    assert "candidate" in cands[0].reason


def test_start_may_be_late_when_motion_precedes_start_uncovered():
    sig = make_signal(active_spans=[(2, 6)])
    flags = signal_flags([ann(3.5, 6, id="late")], sig, 10.0)
    assert "BD_START_MAY_BE_LATE" in codes(flags, "late")
    # covered by a neighbouring annotation -> no flag
    flags2 = signal_flags([ann(2, 3.5, id="prev"), ann(3.5, 6, id="late")], sig, 10.0)
    assert "BD_START_MAY_BE_LATE" not in codes(flags2, "late")


def test_abrupt_change_inside_segment_flagged_but_not_at_boundary():
    sig = make_signal(active_spans=[(0, 10)], cut_at=[5.0])
    assert "BD_ABRUPT_CHANGE" in codes(signal_flags([ann(0, 10, id="a")], sig, 10.0))
    assert "BD_ABRUPT_CHANGE" not in codes(signal_flags([ann(0, 5, id="a"), ann(5, 10, id="b")], sig, 10.0))


def test_model_flags_ignore_stale_verdicts():
    a = ann(0, 2, "grasp the red cube", id="a")
    verdicts = {"a": {"verdict": "contradicted", "observed": "releases cube", "explanation": "x",
                      "model": "m", "snapshot": snapshot(a)}}
    fl = model_flags([a], verdicts)
    assert fl and fl[0].code == "MODEL_CONTRADICTED" and fl[0].origin == FlagOrigin.model
    a.action = "release the red cube"  # reviewer edit -> verdict no longer applies
    assert model_flags([a], verdicts) == []


def test_priority_is_transparent_weighted_count_and_ignores_resolved():
    f1 = Flag(code="X", category="timestamp", severity=Severity.high, origin=FlagOrigin.rule, reason="r",
              annotation_id="a")
    f2 = Flag(code="Y", category="timestamp", severity=Severity.low, origin=FlagOrigin.rule, reason="r",
              annotation_id="a")
    assert priority_score([f1, f2]) == 4
    f1.resolved = True
    assert priority_score([f1, f2]) == 1


def test_rank_orders_by_priority_then_time():
    a, b, c = ann(0, 1, id="a"), ann(1, 2, id="b"), ann(5, 4, id="c")
    flags = run_qc([a, b, c], 10.0)
    assert rank_annotations([a, b, c], flags)[0][0].id == "c"


def test_resolutions_survive_recheck():
    anns = [ann(0, 5, id="a"), ann(4, 8, id="b")]
    first = run_qc(anns, 10.0)
    first[0].resolved, first[0].resolution_note = True, "ok"
    again = carry_resolutions(run_qc(anns, 10.0), first)
    match = [f for f in again if f.key == first[0].key][0]
    assert match.resolved and match.resolution_note == "ok"


# ---------------------------------------------------------------------- BD_START_IN_IDLE (initial idle)
def test_segment_starting_in_initial_idle_is_flagged_not_trimmed():
    sig = make_signal([(1.6, 4.0)])                      # robot still until ~1.6s
    a = ann(0.0, 4.0, "grasp the brick", id="a1")
    a.boundary_precision_s = 0.5                         # frame-sampled vision-model output
    flags = signal_flags([a], sig, 10.0)
    lead = [f for f in flags if f.code == "BD_START_IN_IDLE"]
    assert len(lead) == 1 and 1.4 <= lead[0].details["suggested_start_s"] <= 1.8
    assert (a.start, a.end) == (0.0, 4.0)                # never changed automatically
    assert "EV_IDLE_INSIDE" not in codes(flags, "a1")    # same idle stretch is not reported twice


def test_initial_idle_within_stated_precision_is_not_flagged():
    sig = make_signal([(1.0, 4.0)])
    a = ann(0.0, 4.0, "grasp the brick", id="a1")
    a.boundary_precision_s = 1.5
    assert "BD_START_IN_IDLE" not in codes(signal_flags([a], sig, 10.0))
    short = ann(0.0, 4.0, "grasp the brick", id="a2")    # idle 0.4s < 0.5s minimum
    short.boundary_precision_s = 0.3
    assert "BD_START_IN_IDLE" not in codes(signal_flags([short], make_signal([(0.4, 4.0)]), 10.0))


def test_mid_video_pause_is_not_an_initial_idle_flag():
    sig = make_signal([(0.4, 2.0), (4.0, 8.0)])          # robot already moved before this segment
    a = ann(2.2, 8.0, "place the brick", id="a1")
    a.boundary_precision_s = 0.5
    assert "BD_START_IN_IDLE" not in codes(signal_flags([a], sig, 10.0))


def test_initial_idle_rule_only_applies_to_frame_sampled_model_output():
    sig = make_signal([(1.6, 4.0)])                      # e.g. human / simulated annotation, no stated precision
    assert "BD_START_IN_IDLE" not in codes(signal_flags([ann(0.0, 4.0, "reach", id="a1")], sig, 10.0))


def test_accepting_suggested_start_clears_the_flag(meta):
    from robotrace import review_service as rs
    from robotrace.schemas import ProcessingInfo, ReviewSession
    sig = make_signal([(1.6, 4.0)])
    a = ann(0.0, 4.0, "grasp the brick", id="a1")
    a.boundary_precision_s = 0.5
    flags = run_qc([a], meta.duration_s, sig)
    flag = next(f for f in flags if f.code == "BD_START_IN_IDLE")
    sess = ReviewSession(video=meta, processing=ProcessingInfo(adapter="muse", is_live_inference=True),
                         baseline=[a.model_copy(deep=True)], annotations=[a.model_copy(deep=True)],
                         initial_flags=[f.model_copy(deep=True) for f in flags], flags=flags, signals=sig.to_dict())
    rs.edit_annotation(sess, "a1", start=flag.details["suggested_start_s"])
    assert "BD_START_IN_IDLE" not in codes(sess.flags, "a1")
