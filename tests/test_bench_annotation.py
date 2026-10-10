"""Offline tests for scripts/bench_annotation.py (no network, no model calls)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts" / "bench_annotation.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def pick_episode(n=100, close=40, open_=80):
    """5 fps: still 0-0.8s, moving after; gripper closes at frame `close`, opens at `open_`; lift 2 frames after close."""
    t = np.arange(n) / 5.0
    s = np.zeros((n, 8))
    s[5:, 0] = np.cumsum(np.full(n - 5, 0.01))      # 0.05 units/s motion
    s[95:, 0] = s[94, 0]                            # stops at the end
    s[close:open_, 7] = 1.0
    s[close + 2:open_, 2] = 0.05
    return t, s


def test_ur5_pick_events_are_measured_and_inferred():
    out = bench.ur5_events(*pick_episode())
    e = out["events"]
    assert (e["gripper_close"], e["gripper_open"], e["lift_onset"]) == (8.0, 16.0, 8.4)
    assert e["motion_start"] == 1.0 and e["motion_end"] == 18.8
    assert out["provenance"]["gripper_close"].startswith("measured")
    assert out["provenance"]["lift_onset"].startswith("inferred") and out["n_gripper_toggles"] == 2


def test_ur5_sweep_marks_events_not_applicable_instead_of_inventing_them():
    t = np.arange(70) / 5.0
    s = np.zeros((70, 8))
    s[:, 0] = np.cumsum(np.full(70, 0.012))          # always moving, gripper never toggles
    out = bench.ur5_events(t, s)
    assert all(v is None for v in out["events"].values())
    assert any("not applicable" in n for n in out["notes"]) and any("not measurable" in n for n in out["notes"])
    rows = bench.draft_rows("ur5", 0, "sweep the cloth", out["events"], 14.0)
    assert len(rows) == 1 and rows[0]["draft_action"] == "sweep the cloth" and rows[0]["verified"] == ""


def test_ur5_reference_uses_only_event_bounded_intervals():
    e = bench.ur5_events(*pick_episode())["events"]
    ref = bench.ur5_reference(e, 20.0)
    assert ref["phases"]["approach"] == [1.0, 8.0] and ref["phases"]["transport"] == [8.0, 16.0]
    assert ref["phases"]["return"] == [16.0, 18.8]
    assert ref["phases"]["grasp"] is None and ref["phases"]["release"] is None   # instants at 5 fps, not invented
    assert ref["events"]["grasp"] == 8.0 and ref["events"]["release"] == 16.0
    m = bench.ev.score_episode(ref, [{"start": 1.0, "end": 8.0, "action": "reach toward the cup"}])
    assert m["per_phase"]["approach"]["covered"] and m["timing"]["release"]["abs_error_s"] is None


def test_draft_rows_are_unverified_scaffold():
    e = bench.ur5_events(*pick_episode())["events"]
    rows = bench.draft_rows("ur5", 7, "pick up the blue cup", e, 20.0)
    assert [r["draft_action"] for r in rows] == ["reach toward the object", "grasp the object",
                                                 "lift and move the object", "release and retract"]
    assert all(r["verified"] == "" and r["corrected_action"] == "" for r in rows)


def test_selection_rule_is_deterministic_and_excludes_small_tasks():
    eps = pd.DataFrame({"episode_index": list(range(30)) + [100, 101],
                        "task_index": [0] * 10 + [1] * 10 + [2] * 10 + [3, 3]})
    tasks = {0: "a", 1: "b", 2: "c", 3: "rare"}
    sel = bench.select_ur5(eps, tasks)
    assert sel == bench.select_ur5(eps, tasks)
    assert sel[0] == {"dataset": "ur5", "episode": 4, "split": "regression"}
    dev = [s["episode"] for s in sel if s["split"] == "dev"]
    held = [s["episode"] for s in sel if s["split"] == "heldout"]
    assert dev == [0, 10, 20] and held == [5, 15, 25] and 100 not in dev + held and 4 not in dev


def test_cost_model_matches_observed_runs():
    assert bench.est_input_tokens(24) == 10021 and bench.est_input_tokens(6) == 2821
