"""Generate the bundled SYNTHETIC sample: a rendered 2-link robot arm doing pick-and-place.

Outputs (data/sample/):
  robotrace_sample.mp4        H.264 video (rendered here, so no third-party licence applies)
  ground_truth.json           exact action segments known from the renderer
  baseline_annotations.json   a SIMULATED imperfect annotator output with deliberately injected,
                              documented errors (used to demonstrate and measure QC)

Run:  python scripts/make_sample.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robotrace.video_processor import make_browser_playable, sha256_file  # noqa: E402

OUT = ROOT / "data" / "sample"
W, H, FPS = 640, 360, 15
TABLE_Y = 300
BASE = np.array([100.0, 300.0])
L1, L2 = 240.0, 220.0
FINGER = 34.0          # wrist-to-fingertip distance
CUBE = 36
HOME = np.array([300.0, 110.0])
RED0 = np.array([280.0, TABLE_Y - CUBE / 2])
BLUE0 = np.array([470.0, TABLE_Y - CUBE / 2])
BIN_X, BIN_W, BIN_H = 390.0, 74.0, 52.0
ABOVE = lambda p, dy=0: np.array([p[0], p[1] - CUBE / 2 - FINGER + 14 - dy])  # noqa: E731

# Keyframes: (time_s, wrist_xy, gripper_open[0..1], holding)  holding in {None,"red","blue"}
KEYS = [
    (0.0, HOME, 1.0, None),
    (1.5, HOME, 1.0, None),
    (4.5, ABOVE(RED0), 1.0, None),
    (5.5, ABOVE(RED0), 0.35, "red"),
    (7.5, ABOVE(RED0, 110), 0.35, "red"),
    (11.0, np.array([BIN_X, ABOVE(RED0, 110)[1]]), 0.35, "red"),
    (12.0, np.array([BIN_X, ABOVE(RED0, 110)[1]]), 1.0, None),
    (14.5, HOME, 1.0, None),
    (16.0, HOME, 1.0, None),
    (19.0, ABOVE(BLUE0), 1.0, None),
    (20.0, ABOVE(BLUE0), 0.35, "blue"),
    (22.0, ABOVE(BLUE0 + np.array([60.0, 0])), 0.35, "blue"),
    (23.0, ABOVE(BLUE0 + np.array([60.0, 0])), 1.0, None),
    (25.0, HOME, 1.0, None),
]
DURATION = 25.0
GLITCH = (15.0, 15.2)  # brief lighting glitch (unexplained abrupt change) during idle

GROUND_TRUTH = [
    (1.5, 4.5, "reach toward the red cube"),
    (4.5, 5.5, "grasp the red cube"),
    (5.5, 7.5, "lift the red cube"),
    (7.5, 11.0, "move the red cube over the bin"),
    (11.0, 12.0, "release the red cube into the bin"),
    (12.0, 14.5, "return arm to home position"),
    (16.0, 19.0, "reach toward the blue cube"),
    (19.0, 20.0, "grasp the blue cube"),
    (20.0, 22.0, "slide the blue cube to the right"),
    (22.0, 23.0, "release the blue cube"),
    (23.0, 25.0, "return arm to home position"),
]

# Simulated annotator output. Each entry: (start, end, action, injected_error or None)
BASELINE = [
    (0.2, 1.4, "wipe the table with a sponge", "hallucinated_action"),
    (1.5, 4.5, "reach toward the red cube", None),
    (4.5, 5.5, "grasp the red cube", None),
    (5.5, 7.5, "lift the red cube", None),
    (8.6, 11.0, "move the red cube over the bin", "late_start_boundary"),
    (11.0, 12.0, "pick up the red cube", "wrong_label"),
    (12.0, 17.2, "return arm to home position", "overlong_segment"),
    (16.0, 19.0, "reach toward the blue cube", None),
    # "grasp the blue cube" (19.0-20.0) deliberately omitted -> missing action
    (20.0, 22.0, "slide the blue cube to the right", None),
    (23.0, 22.0, "release the blue cube", "reversed_timestamps"),
    (23.0, 26.5, "return arm to home position", "out_of_range_end"),
]
MISSING = [(19.0, 20.0, "grasp the blue cube")]


def ease(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def state_at(t: float):
    for (t0, p0, g0, h0), (t1, p1, g1, _h1) in zip(KEYS, KEYS[1:]):
        if t0 <= t <= t1:
            u = ease((t - t0) / (t1 - t0)) if t1 > t0 else 1.0
            return p0 + (p1 - p0) * u, g0 + (g1 - g0) * u, h0
    return KEYS[-1][1], KEYS[-1][2], KEYS[-1][3]


def ik(target: np.ndarray):
    d = target - BASE
    r = min(np.linalg.norm(d), L1 + L2 - 1e-3)
    c2 = (r * r - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    q2 = math.acos(max(-1.0, min(1.0, c2)))  # elbow-up in image coordinates (y points down)
    q1 = math.atan2(d[1], d[0]) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    elbow = BASE + L1 * np.array([math.cos(q1), math.sin(q1)])
    wrist = elbow + L2 * np.array([math.cos(q1 + q2), math.sin(q1 + q2)])
    return elbow, wrist


def cube_positions(t: float):
    """Physically consistent cube positions given the scripted gripper state."""
    red, blue = RED0.copy(), BLUE0.copy()
    red_floor = np.array([BIN_X, TABLE_Y - CUBE / 2 - 4])
    for tt in np.arange(0, t + 1e-9, 1 / FPS):
        wrist, _g, holding = state_at(tt)
        if holding == "red":
            red = np.array([wrist[0], wrist[1] + CUBE / 2 + FINGER - 14])
        elif holding == "blue":
            blue = np.array([wrist[0], wrist[1] + CUBE / 2 + FINGER - 14])
        elif tt > 11.5 and red[1] < red_floor[1]:  # fall into bin after release
            red = np.array([BIN_X, min(red[1] + 18, red_floor[1])])
    return red, blue


def render(t: float) -> np.ndarray:
    img = np.full((H, W, 3), (236, 232, 226), np.uint8)
    for x in range(0, W, 40):
        cv2.line(img, (x, 0), (x, TABLE_Y), (228, 224, 218), 1)
    cv2.rectangle(img, (0, TABLE_Y), (W, H), (120, 132, 146), -1)
    cv2.line(img, (0, TABLE_Y), (W, TABLE_Y), (70, 80, 92), 2)
    # bin
    bx0, bx1 = int(BIN_X - BIN_W / 2), int(BIN_X + BIN_W / 2)
    cv2.rectangle(img, (bx0, TABLE_Y - int(BIN_H)), (bx1, TABLE_Y), (200, 205, 210), -1)
    red, blue = cube_positions(t)
    for pos, col in ((red, (40, 40, 210)), (blue, (200, 110, 30))):
        x, y = int(pos[0]), int(pos[1])
        cv2.rectangle(img, (x - CUBE // 2, y - CUBE // 2), (x + CUBE // 2, y + CUBE // 2), col, -1)
        cv2.rectangle(img, (x - CUBE // 2, y - CUBE // 2), (x + CUBE // 2, y + CUBE // 2), (30, 30, 30), 1)
    # bin front wall drawn over cubes inside it
    cv2.rectangle(img, (bx0, TABLE_Y - int(BIN_H)), (bx1, TABLE_Y), (90, 96, 104), 3)
    # arm
    wrist_target, grip, _ = state_at(t)
    elbow, wrist = ik(wrist_target)
    cv2.rectangle(img, (int(BASE[0]) - 34, TABLE_Y - 16), (int(BASE[0]) + 34, TABLE_Y), (60, 64, 70), -1)
    for a, b, th in ((BASE, elbow, 18), (elbow, wrist, 14)):
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (50, 140, 240), th, cv2.LINE_AA)
    for p, r in ((BASE, 14), (elbow, 11), (wrist, 9)):
        cv2.circle(img, tuple(p.astype(int)), r, (40, 44, 50), -1, cv2.LINE_AA)
    half = 8 + grip * 18
    wx, wy = wrist
    cv2.line(img, (int(wx - half), int(wy)), (int(wx + half), int(wy)), (40, 44, 50), 5)
    for s in (-1, 1):
        x = int(wx + s * half)
        cv2.line(img, (x, int(wy)), (x, int(wy + FINGER)), (40, 44, 50), 5)
    cv2.putText(img, f"SYNTHETIC SAMPLE  t={t:5.2f}s", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1,
                cv2.LINE_AA)
    if GLITCH[0] <= t < GLITCH[1]:
        img = cv2.convertScaleAbs(img, alpha=0.45, beta=0)  # lighting drop
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n = int(DURATION * FPS)
    for i in range(n):
        writer.write(render(i / FPS))
    writer.release()
    final = OUT / "robotrace_sample.mp4"
    out = make_browser_playable(raw, final)
    if out == raw:
        raw.replace(final)
        print("WARNING: ffmpeg unavailable; sample stored as mp4v (may not play in every browser)")
    else:
        raw.unlink()
    sha = sha256_file(final)
    common = {"video_file": final.name, "video_sha256": sha, "time_unit": "seconds",
              "synthetic": True, "generator": "scripts/make_sample.py"}
    gt = {**common, "description": "Exact action segments from the renderer's script (ground truth).",
          "segments": [{"start": s, "end": e, "action": a} for s, e, a in GROUND_TRUTH],
          "non_action_events": [{"start": GLITCH[0], "end": GLITCH[1], "event": "lighting glitch"}]}
    base = {**common,
            "description": ("SIMULATED annotator output for demonstration. It is NOT produced by a model: it is "
                            "derived from the ground truth with deliberately injected errors listed in "
                            "`injected_error` and `missing_segments`, so QC flags can be evaluated."),
            "annotations": [{"id": f"ann_s{i:02d}", "start": s, "end": e, "action": a, "injected_error": err}
                            for i, (s, e, a, err) in enumerate(BASELINE)],
            "missing_segments": [{"start": s, "end": e, "action": a} for s, e, a in MISSING]}
    (OUT / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    (OUT / "baseline_annotations.json").write_text(json.dumps(base, indent=2))
    print(f"wrote {final} ({final.stat().st_size / 1e3:.0f} KB, sha256 {sha[:12]}...)")


if __name__ == "__main__":
    main()
