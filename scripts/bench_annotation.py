"""Prepare a small annotation benchmark from SO101 + Berkeley UR5 (no model calls), and score cached predictions.

    python scripts/bench_annotation.py --out out/bench_annotation_<stamp>            # prepare only
    python scripts/bench_annotation.py --out out/bench_annotation_<stamp> --baseline --live   # paid; needs approval

Two tiers of reference, never mixed:
  1. Signal-anchored EVENTS (scored automatically):
     UR5   gripper close / open  = transitions of observation.state[7] (binary; MEASURED, but the field's meaning is
                                   inferred from the data, not documented)
           lift onset            = first frame after close where state[2] rises > UR5_LIFT_DZ above its value at
                                   close (INFERRED: state[0:3] behave like end-effector xyz; not documented)
           motion start / end    = end-effector speed (state[0:3]) above UR5_SPEED_THR for >= UR5_MIN_RUN frames
                                   (INFERRED). Not measurable for the sweeping task (the arm never stops).
     SO101 eval_so101.derive_reference (gripper stall rule; INFERRED from measured signals).
  2. DRAFT segment labels (action text, objects, start/end) built from those events and the task text. They are a
     scaffold for a human to verify in verification_sheet.csv; description/object accuracy is NOT scored until a
     human has marked rows verified.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("eval_so101", ROOT / "scripts" / "eval_so101.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

UR5_REPO = "lerobot/berkeley_autolab_ur5"
UR5_BASE = f"https://huggingface.co/datasets/{UR5_REPO}/resolve/main"
UR5_FPS = 5
CAMERA = {"ur5": "image", "so101": "side"}  # RGB scene cameras (UR5 image_with_depth is a false-colour depth map)
UR5_SPEED_THR = 0.01   # end-effector speed (state units / s); idle frames are ~0.0001, motion ~0.05-0.08
UR5_MIN_RUN = 3        # frames (0.6 s at 5 fps)
UR5_LIFT_DZ = 0.02     # state[2] rise after close that counts as lift onset
UR5_MIN_EPISODES_FOR_SPLIT = 10
UR5_REGRESSION_EPISODE = 4          # the previously working blue-cup clip
SO101_DEV = [1, 3, 4]               # SO101 episodes with merged phases in the original run
SO101_HELDOUT = [20, 30, 40]        # unseen: every 10th episode outside 0-9
SELECTION_RULE = ("UR5: for each task with >= 10 episodes, sort its episode indices; development = the smallest "
                  f"index (excluding regression episode {UR5_REGRESSION_EPISODE}), held-out = the index at position "
                  f"len//2; plus regression episode {UR5_REGRESSION_EPISODE}. Tasks with fewer episodes cannot be "
                  f"split and are excluded. SO101: development = {SO101_DEV} (merged phases in the original run), "
                  f"held-out = {SO101_HELDOUT} (every 10th unseen episode).")


def est_input_tokens(n_frames: int) -> int:
    """Fitted on all 11 observed Muse runs (exact on each): 421 + 400 x frames per request."""
    return 421 + 400 * n_frames


# ------------------------------------------------------------------------------- selection
def select_ur5(eps, task_text: dict) -> list[dict]:
    out = [{"dataset": "ur5", "episode": UR5_REGRESSION_EPISODE, "split": "regression"}]
    for ti, text in sorted(task_text.items()):
        idx = sorted(int(e) for e in eps[eps.task_index == ti].episode_index)
        if len(idx) < UR5_MIN_EPISODES_FOR_SPLIT:
            continue
        dev = next(i for i in idx if i != UR5_REGRESSION_EPISODE)
        held = idx[len(idx) // 2]
        out += [{"dataset": "ur5", "episode": dev, "split": "dev"}, {"dataset": "ur5", "episode": held, "split": "heldout"}]
    return out


# ------------------------------------------------------------------------------- UR5 events
def _runs(mask):
    return ev._runs(np.asarray(mask, bool))


def ur5_events(t: np.ndarray, state: np.ndarray) -> dict:
    g = state[:, 7]
    toggles = np.flatnonzero(np.diff(g) != 0) + 1
    closes = [int(i) for i in toggles if g[i] > g[i - 1]]
    opens = [int(i) for i in toggles if g[i] < g[i - 1]]
    speed = np.r_[0.0, np.linalg.norm(np.diff(state[:, :3], axis=0), axis=1) * UR5_FPS]
    runs = [r for r in _runs(speed > UR5_SPEED_THR) if r[1] - r[0] >= UR5_MIN_RUN]
    always_moving = bool(np.mean(speed[1:] > UR5_SPEED_THR) > 0.95)
    e = {"gripper_close": None, "gripper_open": None, "lift_onset": None, "motion_start": None, "motion_end": None}
    prov = {}
    if closes:
        e["gripper_close"] = float(t[closes[0]]); prov["gripper_close"] = "measured (state[7] 0->1)"
        later = [o for o in opens if o > closes[0]]
        if later:
            e["gripper_open"] = float(t[later[0]]); prov["gripper_open"] = "measured (state[7] 1->0)"
        end = later[0] if later else len(t)
        z0 = state[closes[0], 2]
        up = np.flatnonzero(state[closes[0]:end, 2] - z0 > UR5_LIFT_DZ)
        if up.size:
            e["lift_onset"] = float(t[closes[0] + int(up[0])]); prov["lift_onset"] = "inferred (state[2] rise)"
    if runs and not always_moving:
        e["motion_start"], e["motion_end"] = float(t[runs[0][0]]), float(t[runs[-1][1] - 1])
        prov["motion_start"] = prov["motion_end"] = "inferred (end-effector speed)"
    notes = []
    if not closes:
        notes.append("gripper never toggles: grasp / release / lift not applicable")
    if always_moving:
        notes.append("arm moves throughout: motion start / end not measurable")
    return {"events": {k: None if v is None else round(v, 2) for k, v in e.items()}, "provenance": prov,
            "notes": notes, "n_gripper_toggles": int(len(toggles))}


def ur5_reference(events: dict, duration: float) -> dict:
    """Score-compatible reference: only intervals bounded by events (grasp/release are instants at 5 fps)."""
    ev_ = events
    ph = {p: None for p in ev.PHASES}
    c, o, ms, me = ev_["gripper_close"], ev_["gripper_open"], ev_["motion_start"], ev_["motion_end"]
    if c is not None and ms is not None and ms < c:
        ph["approach"] = [ms, c]
    if c is not None and o is not None:
        ph["transport"] = [c, o]
    if o is not None and me is not None and me > o:
        ph["return"] = [o, me]
    return {"duration_s": duration, "phases": ph,
            "events": {"grasp": c, "release": o, "motion_start": ms, "motion_end": me}, "notes": []}


def draft_rows(dataset: str, ep: int, task: str, events: dict, duration: float) -> list[dict]:
    """Template scaffold for human verification (NOT ground truth)."""
    e = events
    base = {"dataset": dataset, "episode": ep, "task": task, "verified": "", "corrected_action": "",
            "corrected_objects": "", "corrected_start_s": "", "corrected_end_s": "", "notes": ""}
    c, o = e.get("gripper_close", e.get("grasp")), e.get("gripper_open", e.get("release"))
    ms, me, lift = e.get("motion_start"), e.get("motion_end"), e.get("lift_onset")
    if c is None:
        return [{**base, "draft_action": task, "draft_start_s": ms if ms is not None else 0.0,
                 "draft_end_s": me if me is not None else round(duration, 2), "draft_basis": "whole task (no gripper events)"}]
    rows = []
    if ms is not None and ms < c:
        rows.append({"draft_action": "reach toward the object", "draft_start_s": ms, "draft_end_s": c,
                     "draft_basis": "motion start -> gripper close"})
    hold_end = o if o is not None else (me if me is not None else round(duration, 2))
    if lift is not None and lift < hold_end:
        rows.append({"draft_action": "grasp the object", "draft_start_s": c, "draft_end_s": lift,
                     "draft_basis": "gripper close -> lift onset"})
        rows.append({"draft_action": "lift and move the object", "draft_start_s": lift, "draft_end_s": hold_end,
                     "draft_basis": "lift onset -> gripper open"})
    else:
        rows.append({"draft_action": "grasp and move the object", "draft_start_s": c, "draft_end_s": hold_end,
                     "draft_basis": "gripper close -> gripper open"})
    if o is not None and me is not None and me > o:
        rows.append({"draft_action": "release and retract", "draft_start_s": o, "draft_end_s": me,
                     "draft_basis": "gripper open -> motion end"})
    return [{**base, **r} for r in rows]


# ------------------------------------------------------------------------------- contact sheets
def contact_sheet(clip: Path, out: Path, marks: dict, step_s: float = 1.0, cols: int = 6, width: int = 240) -> None:
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tiles = []
    for t in np.arange(0, n / fps, step_s):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (width, int(f.shape[0] * width / f.shape[1])))
        label = f"t={t:.1f}s " + " ".join(k.upper() for k, v in marks.items() if v is not None and abs(v - t) < step_s / 2)
        cv2.rectangle(f, (0, 0), (width, 18), (0, 0, 0), -1)
        cv2.putText(f, label[:34], (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(f)
    cap.release()
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="fresh benchmark directory under out/")
    ap.add_argument("--baseline", action="store_true", help="score cached predictions (with --live: make them)")
    ap.add_argument("--live", action="store_true", help="paid Muse calls (one per clip, cached); needs approval")
    ap.add_argument("--max-calls", type=int, default=0)
    args = ap.parse_args(argv)
    args.out = args.out.resolve()
    import pyarrow.parquet as pq
    from robotrace.video_processor import probe

    cache = args.out / "cache"
    # ---- UR5
    u = cache / "ur5"
    for f in ("README.md", "meta/info.json", "meta/tasks.parquet", "meta/episodes/chunk-000/file-000.parquet",
              "data/chunk-000/file-000.parquet"):
        ev.cached(u / f, lambda f=f: ev.http_get(f"{UR5_BASE}/{f}"))
    u_info = json.loads((u / "meta/info.json").read_text(encoding="utf-8"))
    assert u_info["fps"] == UR5_FPS, u_info["fps"]
    u_tasks = dict(enumerate(pq.read_table(u / "meta/tasks.parquet").to_pandas().index))
    u_eps = pq.read_table(u / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
    u_eps["task_index"] = u_eps["tasks"].apply(lambda x: next(k for k, v in u_tasks.items() if v == x[0]))
    u_data = pq.read_table(u / "data/chunk-000/file-000.parquet").to_pandas()
    selected = select_ur5(u_eps, u_tasks)
    # ---- SO101 (own cache; the frozen out/eval_so101 is never written)
    s = cache / "so101"
    s_lic = ev.dataset_license(s)
    s_info, s_tasks, s_eps, s_data = ev.load_metadata(s)
    selected += [{"dataset": "so101", "episode": e, "split": "dev"} for e in SO101_DEV]
    selected += [{"dataset": "so101", "episode": e, "split": "heldout"} for e in SO101_HELDOUT]

    licences = {"ur5": next((l.split(":", 1)[1].strip() for l in (u / "README.md").read_text(encoding="utf-8").splitlines()
                             if l.startswith("license:")), "unknown"), "so101": s_lic}
    tables, entries, sheet = {}, [], []
    for sel in selected:
        ds, ep = sel["dataset"], sel["episode"]
        cam = CAMERA[ds]
        key = f"videos/observation.images.{cam}"
        if ds == "ur5":
            row = u_eps[u_eps.episode_index == ep].iloc[0]
            fi = int(row[f"{key}/file_index"])
            tables.setdefault((ds, fi), ev.video_index(u, cam, base=UR5_BASE, file_index=fi))
            clip = ev.prepare_clip(u, cam, tables[(ds, fi)], ep, float(row[f"{key}/from_timestamp"]),
                                   float(row[f"{key}/to_timestamp"]), int(row["length"]), fps=UR5_FPS, prefix="ur5")
            d = u_data[u_data.episode_index == ep].sort_values("frame_index")
            st = np.stack(d["observation.state"].to_numpy()).astype(float)
            sig = ur5_events(d.timestamp.to_numpy(float), st)
            task = u_tasks[int(row["task_index"])]
            dur = int(row["length"]) / UR5_FPS
            ref = ur5_reference(sig["events"], dur)
            events, prov, notes = sig["events"], sig["provenance"], sig["notes"]
        else:
            row = s_eps[s_eps.episode_index == ep].iloc[0]
            tables.setdefault((ds, 0), ev.video_index(s, cam))
            clip = ev.prepare_clip(s, cam, tables[(ds, 0)], ep, float(row[f"{key}/from_timestamp"]),
                                   float(row[f"{key}/to_timestamp"]), int(row["length"]))
            d = s_data[s_data.episode_index == ep].sort_values("frame_index")
            ref = ev.derive_reference(d.timestamp.to_numpy(np.float64),
                                      np.stack(d["observation.state"].to_numpy()).astype(np.float64),
                                      np.stack(d["action"].to_numpy()).astype(np.float64))
            task = s_tasks.reset_index().iloc[0]["index"]
            dur = ref["duration_s"]
            events = {"grasp": ref["events"]["grasp"], "release": ref["events"]["release"],
                      "motion_start": ref["events"]["motion_start"], "motion_end": ref["events"]["motion_end"]}
            prov = {k: "inferred (SO101 gripper-stall / joint-speed rules)" for k, v in events.items() if v is not None}
            notes = ref["notes"]
        m = probe(clip)
        assert abs(m.duration_s - dur) < 0.15, (ds, ep, m.duration_s, dur)
        n_frames = int(min(24, max(4, round(m.duration_s))))
        sheet_img = args.out / "contact_sheets" / f"{ds}_ep{ep:03d}.jpg"
        contact_sheet(clip, sheet_img, events)
        entry = {**sel, "task": task, "camera": cam, "duration_s": round(m.duration_s, 2), "fps": m.fps,
                 "clip": str(clip.relative_to(ROOT)), "contact_sheet": str(sheet_img.relative_to(ROOT)),
                 "events": events, "event_provenance": prov, "notes": notes, "reference_for_scoring": ref,
                 "baseline_plan": {"sampling": "legacy", "frames": n_frames, "requests": 1,
                                   "est_input_tokens": est_input_tokens(n_frames)}}
        entries.append(entry)
        sheet += draft_rows(ds, ep, task, events, m.duration_s)
        print(f"{ds} ep{ep:<4} {sel['split']:10s} {m.duration_s:5.1f}s  events {events}  {notes or ''}")

    bench = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "selection_rule": SELECTION_RULE,
             "licences": licences, "cameras": CAMERA,
             "thresholds": {"UR5_SPEED_THR": UR5_SPEED_THR, "UR5_MIN_RUN": UR5_MIN_RUN, "UR5_LIFT_DZ": UR5_LIFT_DZ,
                            "SO101": ev.REF_RULES},
             "label_status": "draft scaffold only; no human verification yet",
             "episodes": entries,
             "baseline_cost_estimate": {"calls": len(entries),
                                        "est_input_tokens": sum(e["baseline_plan"]["est_input_tokens"] for e in entries)}}
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "benchmark.json"
    if target.exists():
        target = args.out / f"benchmark_{time.strftime('%Y%m%d-%H%M%S')}.json"
    target.write_text(json.dumps(bench, indent=2, default=str), encoding="utf-8")
    vs = args.out / "verification_sheet.csv"
    if not vs.exists():  # never overwrite a sheet a human may have edited
        with open(vs, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sheet[0].keys()))
            w.writeheader()
            w.writerows(sheet)
    print(f"\n{len(entries)} clips; baseline estimate: {bench['baseline_cost_estimate']}; wrote {target.name}")
    if args.baseline:
        print("Baseline scoring/calls are not implemented in this preparation step; see the approval request.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
