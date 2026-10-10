"""Evaluate RoboTrace's Muse Spark annotations on lerobot/svla_so101_pickplace against signal-derived references.

    python scripts/eval_so101.py                      # prepare data + references only (no API calls)
    python scripts/eval_so101.py --live               # also run Muse Spark (<= 1 call per episode, cached)
    python scripts/eval_so101.py --episodes 0-9 --camera side --live

What the reference is (and is not)
----------------------------------
The dataset records, at 30 fps, measured joint/gripper positions (`observation.state`) and the operator's
commanded positions (`action`). It does NOT contain human-annotated action boundaries. Reference phases are
DERIVED from those signals with the fixed rules below (REF_RULES); they are rule-based, not ground truth.

  measured     timestamps, joint positions, gripper position, gripper command
  rule events  motion_start / motion_end   first / last frame of sustained (>= MOTION_MIN_RUN_S) joint
                                           speed above MOTION_SPEED_THR
               grasp_t                     start of the longest "holding" run: gripper commanded closed
                                           (cmd <= HOLD_CMD_MAX) but stalled open by >= HOLD_GAP (an object
                                           is blocking it) and nearly stationary
               release_t                   end of that holding run (gripper starts to open)
  phases       approach   [motion_start, close_onset)   close_onset = last frame before grasp_t with
                                                         cmd >= CLOSE_ONSET_FRAC x the max pre-grasp cmd
               grasp      [close_onset, grasp_t)
               transport  [grasp_t, release_t)          lift + transport + lowering are MERGED (no
                                                         end-effector height is recorded)
               release    [release_t, release_end)      release_end = end of the gripper's opening plateau (placement
                                                         and release are MERGED)
               return     [release_end, motion_end)

Predictions are mapped to these phases by the ordered keyword rules in PHASE_RULES (first match wins);
labels matching no rule are reported as "unmapped" and never guessed.

Every live call goes through RoboTrace's own `process_video(..., "muse")`. A cached successful prediction
is always reused; a cached FAILED prediction is not retried unless --retry-failed is given (no surprise
charges). Failed calls are reported as failures and excluded from aggregate metrics.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robotrace.evaluation import Segment, evaluate_segments  # noqa: E402
from robotrace.schemas import Annotation  # noqa: E402

REPO = "lerobot/svla_so101_pickplace"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
OUT = ROOT / "out" / "eval_so101"
FPS = 30
PHASES = ["approach", "grasp", "transport", "release", "return"]

# ------------------------------------------------------------------------------- reference thresholds
REF_RULES = {
    "MOTION_SPEED_THR": 20.0,    # joint speed (norm over 5 arm joints, position units / s, 5-frame mean);
                                 # typical moving speed is ~70, start-up twitches reach ~15
    "MOTION_MIN_RUN_S": 0.3,     # motion must stay above the threshold this long (ignores brief twitches)
    "SMOOTH_FRAMES": 5,
    "HOLD_CMD_MAX": 5.0,         # gripper command at or below this = "commanded closed"
    "HOLD_GAP": 3.0,             # measured - commanded >= this = gripper blocked open by an object
    "HOLD_STILL": 0.5,           # |d gripper position| per frame below this = settled
    "MIN_HOLD_S": 0.3,           # shortest holding run accepted as a grasp
    "CLOSE_ONSET_FRAC": 0.9,     # close_onset: last frame with cmd >= this fraction of the pre-grasp max
    "RELEASE_PEAK_WINDOW_S": 2.0,  # window after release_t searched for the opening peak
    "RELEASE_PEAK_FRAC": 0.95,   # release_end = last frame in that window with gripper >= this x window max
}

# ------------------------------------------------------------------------------- phase mapping (fixed)
# v1: original rules (used for the original 10-episode report).
# v2: `return` also matches "move away ..." and "move the [adjective] arm/gripper/robot back|away" (v1 missed
#     "move away from the clear container" and "move the empty gripper away from the container", mapping them to
#     `approach`). Objects being carried "away" (e.g. "lift the brick away") are deliberately NOT matched.
MAPPING_VERSION = "v2"

PHASE_RULES: list[tuple[str, str]] = [
    ("idle", r"\b(idle|wait(s|ing)?|at rest|resting|stationary|remain(s)? still|no (motion|movement))\b"),
    ("return", r"\b(retract\w*|return\w*|withdraw\w*|retreat\w*|back to|home|mov\w* away|"
               r"mov\w* (the )?(\w+ )?(arm|gripper|robot) (back|away|up and away))\b"),
    ("release", r"\b(release\w*|let go|open\w* (the )?gripper|drop\w*|place\w*|put\w*|deposit\w*|insert\w*|lower\w* .*\b(in|into|inside)\b)"),
    ("transport", r"\b(lift\w*|rais\w*|carr\w*|transport\w*|bring\w*|hold\w*|move\w* (the )?(pink )?(lego )?(brick|block|object|piece))\b"),
    ("grasp", r"\b(grasp\w*|grip(s|ped|ping)?|grab\w*|pick\w*|clos\w* (the )?gripper|clamp\w*|seiz\w*)\b"),
    ("approach", r"\b(reach\w*|approach\w*|mov\w*|extend\w*|position\w*|align\w*|lower\w*|descend\w*|toward\w*)\b"),
]


def map_phase(label: str) -> Optional[str]:
    """Map a free-text action label to a phase name (or 'idle'); None = unmapped."""
    text = label.lower()
    for phase, pattern in PHASE_RULES:
        if re.search(pattern, text):
            return phase
    return None


# ------------------------------------------------------------------------------- downloads + cache
def http_get(url: str, byte_range: Optional[tuple[int, int]] = None, timeout: float = 120) -> bytes:
    import requests
    headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}
    r = requests.get(url, headers=headers, timeout=timeout)
    if byte_range:
        want = byte_range[1] - byte_range[0] + 1
        if r.status_code != 206 or len(r.content) != want:
            raise RuntimeError(f"range request failed: HTTP {r.status_code}, {len(r.content)} of {want} bytes")
    else:
        r.raise_for_status()
    return r.content


def cached(path: Path, fetch: Callable[[], bytes]) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(fetch())
        tmp.replace(path)
    return path


def dataset_license(cache: Path) -> str:
    readme = cached(cache / "README.md", lambda: http_get(f"{BASE}/README.md")).read_text(encoding="utf-8")
    m = re.search(r"^license:\s*(\S+)", readme, re.M)
    return m.group(1) if m else "unknown"


def load_metadata(cache: Path):
    import pyarrow.parquet as pq
    files = {"info": "meta/info.json", "tasks": "meta/tasks.parquet",
             "episodes": "meta/episodes/chunk-000/file-000.parquet", "data": "data/chunk-000/file-000.parquet"}
    paths = {k: cached(cache / v, lambda v=v: http_get(f"{BASE}/{v}")) for k, v in files.items()}
    info = json.loads(paths["info"].read_text(encoding="utf-8"))
    tasks = pq.read_table(paths["tasks"]).to_pandas().reset_index()
    eps = pq.read_table(paths["episodes"]).to_pandas()
    data = pq.read_table(paths["data"]).to_pandas()
    return info, tasks, eps, data


# ------------------------------------------------------------------------------- MP4 / AV1 extraction
_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


def top_level_boxes(buf: bytes) -> list[tuple[bytes, int, int]]:
    out, off = [], 0
    while off + 8 <= len(buf):
        size, typ = struct.unpack(">I4s", buf[off:off + 8])
        if size == 1 and off + 16 <= len(buf):
            size = struct.unpack(">Q", buf[off + 8:off + 16])[0]
        if size < 8:
            break
        out.append((typ, off, size))
        off += size
    return out


def parse_moov(moov: bytes) -> dict:
    """Sample table of the first (video) track: times (s), byte offsets, sizes, keyframes, codec, av1C OBUs."""
    boxes: dict[bytes, bytes] = {}

    def walk(b, off, end):
        while off + 8 <= end:
            size, typ = struct.unpack(">I4s", b[off:off + 8])
            hdr = 8
            if size == 1:
                size, hdr = struct.unpack(">Q", b[off + 8:off + 16])[0], 16
            if size < hdr:
                return
            if typ in _CONTAINERS:
                walk(b, off + hdr, off + size)
            else:
                boxes.setdefault(typ, b[off + hdr:off + size])
            off += size

    walk(moov, 0, len(moov))

    def table(name, fmt, start=8):
        b = boxes[name]
        n = struct.unpack(">I", b[4:8])[0]
        sz = struct.calcsize(fmt)
        return [struct.unpack(fmt, b[start + i * sz:start + (i + 1) * sz]) for i in range(n)]

    mdhd = boxes[b"mdhd"]
    timescale = struct.unpack(">I", mdhd[12:16] if mdhd[0] == 0 else mdhd[20:24])[0]
    times, t = [], 0
    for count, delta in table(b"stts", ">II"):
        for _ in range(count):
            times.append(t / timescale)
            t += delta
    stsz = boxes[b"stsz"]
    fixed, n = struct.unpack(">II", stsz[4:12])
    sizes = [fixed] * n if fixed else [struct.unpack(">I", stsz[12 + 4 * i:16 + 4 * i])[0] for i in range(n)]
    chunks = [o for (o,) in table(b"co64", ">Q")] if b"co64" in boxes else [o for (o,) in table(b"stco", ">I")]
    stsc = table(b"stsc", ">III")
    offsets, s = [], 0
    for ci, co in enumerate(chunks, start=1):
        per = next(spc for first, spc, _ in reversed(stsc) if first <= ci)
        o = co
        for _ in range(per):
            if s >= n:
                break
            offsets.append(o)
            o += sizes[s]
            s += 1
    keyframes = {k - 1 for (k,) in table(b"stss", ">I")} if b"stss" in boxes else set(range(n))
    stsd = boxes[b"stsd"]
    codec = stsd[12:16].decode("latin1")
    width, height = struct.unpack(">HH", stsd[8 + 24 + 8:8 + 24 + 12])
    config = b""
    i = stsd.find(b"av1C")
    if i >= 4:
        size = struct.unpack(">I", stsd[i - 4:i])[0]
        config = stsd[i + 8:i - 4 + size]
    return {"times": times, "offsets": offsets, "sizes": sizes, "keyframes": keyframes, "codec": codec,
            "width": width, "height": height, "config_obus": config}


def episode_span(table: dict, t0: float, t1: float) -> tuple[list[int], int, int]:
    """Samples with t in [t0, t1); raises if decoding would need frames before t0."""
    eps = 1e-6
    inside = [i for i, t in enumerate(table["times"]) if t0 - eps <= t < t1 - eps]
    if not inside:
        raise RuntimeError(f"no video samples in [{t0}, {t1})")
    if inside[0] not in table["keyframes"]:
        raise RuntimeError("episode does not start on a keyframe")
    lo = min(table["offsets"][i] for i in inside)
    hi = max(table["offsets"][i] + table["sizes"][i] for i in inside)
    return inside, lo, hi


def write_ivf(path: Path, frames: list[bytes], width: int, height: int, fps: int) -> None:
    with open(path, "wb") as f:
        f.write(b"DKIF" + struct.pack("<HH4sHHIIII", 0, 32, b"AV01", width, height, fps, 1, len(frames), 0))
        for pts, fr in enumerate(frames):
            f.write(struct.pack("<IQ", len(fr), pts) + fr)


def video_index(cache: Path, camera: str) -> dict:
    url = f"{BASE}/videos/observation.images.{camera}/chunk-000/file-000.mp4"
    head = cached(cache / f"video_{camera}_head.bin", lambda: http_get(url, (0, 65535))).read_bytes()
    moov_box = next(((o, s) for typ, o, s in top_level_boxes(head) if typ == b"moov"), None)
    if moov_box is None:  # moov after mdat: its offset is right after the last box we can see
        last = top_level_boxes(head)[-1]
        if last[0] != b"mdat":
            raise RuntimeError("could not locate the moov box")
        start = last[1] + last[2]
        size_hdr = http_get(url, (start, start + 7))
        moov_box = (start, struct.unpack(">I", size_hdr[:4])[0])
    off, size = moov_box
    moov = cached(cache / f"video_{camera}_moov.bin", lambda: http_get(url, (off, off + size - 1))).read_bytes()
    table = parse_moov(moov[8:])
    table["url"] = url
    return table


def prepare_clip(cache: Path, camera: str, table: dict, ep: int, t0: float, t1: float, n_frames: int) -> Path:
    clip = cache / "clips" / f"so101_ep{ep:03d}_{camera}.mp4"
    if clip.exists():
        return clip
    if table["codec"] != "av01":
        raise RuntimeError(f"unsupported codec {table['codec']}")
    inside, lo, hi = episode_span(table, t0, t1)
    if len(inside) != n_frames:
        raise RuntimeError(f"video has {len(inside)} frames for the episode, metadata says {n_frames}")
    span = cached(cache / "spans" / f"{camera}_ep{ep:03d}.bin", lambda: http_get(table["url"], (lo, hi - 1))).read_bytes()
    frames = []
    for k, s in enumerate(inside):
        a = table["offsets"][s] - lo
        blob = span[a:a + table["sizes"][s]]
        frames.append(table["config_obus"] + blob if k == 0 else blob)
    import imageio_ffmpeg
    clip.parent.mkdir(parents=True, exist_ok=True)
    ivf = clip.with_suffix(".ivf")
    write_ivf(ivf, frames, table["width"], table["height"], FPS)
    tmp = clip.with_name(clip.stem + ".part.mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-c:v", "libdav1d",
                    "-i", str(ivf), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
                    "-movflags", "+faststart", str(tmp)], check=True, timeout=300)
    ivf.unlink(missing_ok=True)
    tmp.replace(clip)
    return clip


# ------------------------------------------------------------------------------- reference intervals
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as [start, end) index pairs."""
    runs, start = [], None
    for i, m in enumerate(np.r_[mask, False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i))
            start = None
    return runs


def derive_reference(t: np.ndarray, state: np.ndarray, action: np.ndarray, rules: dict = REF_RULES) -> dict:
    """Rule-derived events and phases from measured signals. Missing events are None (never invented)."""
    n = len(t)
    joints = state[:, :5]
    speed = np.r_[0.0, np.linalg.norm(np.diff(joints, axis=0), axis=1) * FPS]
    k = rules["SMOOTH_FRAMES"]
    speed = np.convolve(speed, np.ones(k) / k, mode="same")
    min_run = int(round(rules["MOTION_MIN_RUN_S"] * FPS))
    moving_runs = [r for r in _runs(speed > rules["MOTION_SPEED_THR"]) if r[1] - r[0] >= min_run]
    ev = {"motion_start": None, "motion_end": None, "grasp": None, "release": None}
    notes = []
    if moving_runs:
        ev["motion_start"], ev["motion_end"] = float(t[moving_runs[0][0]]), float(t[moving_runs[-1][1] - 1])
    else:
        notes.append("no joint motion above threshold")

    g, cmd = state[:, 5], action[:, 5]
    dg = np.r_[0.0, np.abs(np.diff(g))]
    hold = (cmd <= rules["HOLD_CMD_MAX"]) & (g - cmd >= rules["HOLD_GAP"]) & (dg < rules["HOLD_STILL"])
    runs = [r for r in _runs(hold) if (r[1] - r[0]) / FPS >= rules["MIN_HOLD_S"]]
    phases: dict[str, Optional[list[float]]] = {p: None for p in PHASES}
    if not runs:
        notes.append("no holding run (gripper never stalled on an object): grasp/release not measurable")
    else:
        a, b = max(runs, key=lambda r: r[1] - r[0])
        ev["grasp"] = float(t[a])
        ev["release"] = float(t[min(b, n - 1)])
        pre = cmd[:a]
        onset = a
        if pre.size:
            thr = rules["CLOSE_ONSET_FRAC"] * float(pre.max())
            above = np.flatnonzero(pre >= thr)
            onset = int(above[-1]) if above.size else a
        w_end = min(n, b + int(rules["RELEASE_PEAK_WINDOW_S"] * FPS))
        win = g[b:w_end]  # release_end: last frame of the opening plateau (>= 95% of the window max)
        peak = b + int(np.flatnonzero(win >= rules["RELEASE_PEAK_FRAC"] * win.max())[-1]) if win.size else b
        ms, me = ev["motion_start"], ev["motion_end"]
        if ms is not None and ms < t[onset]:
            phases["approach"] = [ms, float(t[onset])]
        if onset < a:
            phases["grasp"] = [float(t[onset]), ev["grasp"]]
        phases["transport"] = [ev["grasp"], ev["release"]]
        if peak > b:
            phases["release"] = [ev["release"], float(t[peak])]
        rel_end = float(t[peak])
        if me is not None and me > rel_end:
            phases["return"] = [rel_end, me]
    duration = float(t[-1] + 1 / FPS)
    return {"duration_s": round(duration, 3), "events": {k: None if v is None else round(v, 3) for k, v in ev.items()},
            "phases": {k: None if v is None else [round(v[0], 3), round(v[1], 3)] for k, v in phases.items()},
            "notes": notes, "rules": dict(rules)}


# ------------------------------------------------------------------------------- scoring
def _labels_on_grid(intervals: list[tuple[float, float, str]], duration: float, default: str) -> list[str]:
    grid = [default] * int(round(duration * FPS))
    for s, e, lab in intervals:
        for i in range(max(0, int(np.floor(s * FPS))), min(len(grid), int(np.ceil(e * FPS)))):
            grid[i] = lab
    return grid


def _interval_iou(ref: tuple[float, float], preds: list[tuple[float, float]], duration: float) -> float:
    r = np.zeros(int(round(duration * FPS)), bool)
    p = r.copy()
    r[int(ref[0] * FPS):int(np.ceil(ref[1] * FPS))] = True
    for s, e in preds:
        p[max(0, int(s * FPS)):max(0, int(np.ceil(e * FPS)))] = True
    union = (r | p).sum()
    return float((r & p).sum() / union) if union else 0.0


def score_episode(ref: dict, annotations: list[dict]) -> dict:
    """Compare mapped predictions with the rule-derived reference. Unmeasurable metrics are None."""
    dur = ref["duration_s"]
    mapped, unmapped = [], []
    for a in annotations:
        ph = map_phase(a["action"])
        if ph is None:
            unmapped.append(a["action"])
        mapped.append({**a, "phase": ph})
    valid = [m for m in mapped if m["phase"] and m["phase"] != "idle" and m["end"] > m["start"]]

    ref_phases = {p: v for p, v in ref["phases"].items() if v}
    per_phase, covered = {}, 0
    for p, (s, e) in ref_phases.items():
        preds = [(m["start"], m["end"]) for m in valid if m["phase"] == p]
        overlap = any(min(e, pe) > max(s, ps) for ps, pe in preds)
        covered += overlap
        per_phase[p] = {"reference": [s, e], "predicted": preds, "covered": overlap,
                        "iou": round(_interval_iou((s, e), preds, dur), 3) if preds else 0.0}
    coverage = round(covered / len(ref_phases), 3) if ref_phases else None

    def first(phase, key):
        xs = [m[key] for m in valid if m["phase"] == phase]
        return (min(xs) if key == "start" else max(xs)) if xs else None

    pred_grasp = first("grasp", "end")
    if pred_grasp is None:
        pred_grasp = first("transport", "start")
    pred_release = first("release", "start")
    if pred_release is None:
        pred_release = first("transport", "end")
    pred_events = {"grasp": pred_grasp, "release": pred_release,
                   "motion_start": min((m["start"] for m in valid), default=None),
                   "motion_end": max((m["end"] for m in valid), default=None)}
    timing = {}
    for k, r in ref["events"].items():
        p = pred_events[k]
        timing[k] = {"reference_s": r, "predicted_s": p,
                     "abs_error_s": None if r is None or p is None else round(abs(p - r), 3),
                     "measurable": r is not None and p is not None}

    order = [m["phase"] for m in sorted(valid, key=lambda m: (m["start"], m["end"]))]
    seq = [p for i, p in enumerate(order) if i == 0 or p != order[i - 1]]
    rank = [PHASES.index(p) for p in seq]
    pairs = list(zip(rank, rank[1:]))
    order_score = round(sum(b > a for a, b in pairs) / len(pairs), 3) if pairs else None

    ref_grid = _labels_on_grid([(v[0], v[1], p) for p, v in ref_phases.items()], dur, "idle")
    pred_grid = _labels_on_grid([(m["start"], m["end"], m["phase"]) for m in valid], dur, "idle")
    frame_acc = round(sum(a == b for a, b in zip(ref_grid, pred_grid)) / len(ref_grid), 3) if ref_grid else None

    seg = evaluate_segments([Annotation(start=m["start"], end=m["end"], action=m["phase"]) for m in valid],
                            [Segment(v[0], v[1], p) for p, v in ref_phases.items()], iou_threshold=0.3)
    return {"phase_coverage": coverage, "n_reference_phases": len(ref_phases), "per_phase": per_phase,
            "timing": timing, "predicted_phase_sequence": seq, "order_agreement": order_score,
            "order_fully_correct": None if order_score is None else order_score == 1.0,
            "frame_agreement": frame_acc, "segment_match_iou0.3": seg.as_dict(),
            "mapped_predictions": [{k: m[k] for k in ("start", "end", "action", "phase")} for m in mapped],
            "unmapped_predictions": unmapped}


# ------------------------------------------------------------------------------- inference (cached)
# Multi-window runs report "response id a+b+c" and summed tokens; the pattern accepts both forms.
_DETAIL_RE = re.compile(r"response id (?P<rid>[^,\s]+)(?:, tokens in/out (?P<tin>\d+)/(?P<tout>\d+))?")


LEGACY_PROMPT_VERSION = "obs-v1"  # predictions made before prompt versions were recorded


LEGACY_SAMPLING = {"mode": "legacy", "max_frames": 24}  # implied for records made before sampling was recorded


def sampling_tag(cfg: dict) -> str:
    """Folder name for a sampling config; legacy keeps the original layout."""
    if cfg.get("mode") == "legacy":
        return ""
    return f"windowed-{cfg['spacing_s']:g}s-ov{cfg['overlap_s']:g}-max{cfg['max_windows']}x{cfg['max_frames']}"


def predictions_dir(out: Path, prompt_version: str, sampling: Optional[dict] = None,
                    model: Optional[str] = None) -> Path:
    """obs-v1 predictions live in the original flat folder; later prompt versions are separated by model and
    sampling config so results from different configurations can never share (or overwrite) a file."""
    base = out / "predictions"
    if prompt_version == LEGACY_PROMPT_VERSION:
        d = base
    else:
        d = base / prompt_version / re.sub(r"[^A-Za-z0-9._-]", "_", model or "default-model")
    tag = sampling_tag(sampling or LEGACY_SAMPLING)
    return d / tag if tag else d


def run_inference(clip: Path, pred_path: Path, *, live: bool, retry_failed: bool, budget: dict,
                  process: Optional[Callable] = None, prompt_version: Optional[str] = None,
                  sampling: Optional[dict] = None, model: Optional[str] = None) -> dict:
    """Return a prediction record. Reuses cache; at most one live call per episode; never retries silently.

    A cached record is reused only if its model, prompt version and sampling config all match. A live run never
    overwrites an existing cached record it cannot reuse (a paid result is never destroyed).
    """
    if prompt_version is None:
        from robotrace.annotation_engine import OBSERVATION_PROMPT_VERSION as prompt_version
    if sampling is None:
        from robotrace.config import get_settings
        from robotrace.sampling import sampling_config
        sampling = sampling_config(get_settings())
    if model is None:
        from robotrace.config import get_settings
        model = get_settings().muse_model
    if pred_path.exists():
        try:
            rec = json.loads(pred_path.read_text(encoding="utf-8"))
            cached_version = rec.get("prompt_version") or LEGACY_PROMPT_VERSION
            cached_sampling = (rec.get("sampling") or {}).get("config") or LEGACY_SAMPLING
            cached_model = rec.get("model")
            mismatch = (cached_version != prompt_version or cached_sampling != sampling
                        or (cached_model is not None and cached_model != model))
            if rec.get("status") == "ok" and mismatch:
                return {"status": "stale_cache", "error": f"cached prediction used model {cached_model} / prompt "
                                                          f"{cached_version} / sampling {cached_sampling}, expected "
                                                          f"{model} / {prompt_version} / {sampling}"
                                                          + ("; not overwritten" if live else "")}
            elif rec.get("status") == "ok" and isinstance(rec.get("annotations"), list):
                return {**rec, "prompt_version": cached_version, "from_cache": True}
            elif rec.get("status") == "failed" and not retry_failed:
                return {**rec, "from_cache": True}
        except (ValueError, OSError) as e:
            return {"status": "invalid_cache", "error": f"cached prediction unreadable: {type(e).__name__}"
                                                        + ("; not overwritten" if live else "")}
    if not live:
        return {"status": "not_run", "error": "no cached prediction and --live not given"}
    if budget["calls"] >= budget["max_calls"]:
        return {"status": "skipped_budget", "error": f"call budget of {budget['max_calls']} reached"}

    from robotrace.annotation_engine import AdapterUnavailable
    from robotrace.config import get_settings
    from robotrace.pipeline import process_video
    from robotrace.video_processor import probe
    process = process or process_video
    budget["calls"] += 1
    t0 = time.perf_counter()
    try:
        meta = probe(clip, display_name=clip.name)
        sess = process(clip, meta, "muse", settings=get_settings())
        p = sess.processing
        if p.adapter != "muse" or not p.is_live_inference:
            raise RuntimeError(f"unexpected engine result: adapter={p.adapter}, live={p.is_live_inference}")
        if p.prompt_version != prompt_version:
            raise RuntimeError(f"prompt version mismatch: ran {p.prompt_version!r}, expected {prompt_version!r}")
        if (p.sampling or {}).get("config") != sampling:
            raise RuntimeError(f"sampling mismatch: ran {(p.sampling or {}).get('config')}, expected {sampling}")
        m = _DETAIL_RE.search(p.adapter_detail or "")
        rec = {"status": "ok", "provider": p.provider, "model": p.model, "prompt_version": p.prompt_version,
               "sampling": p.sampling,
               "response_id": m.group("rid") if m else None,
               "tokens_in": int(m.group("tin")) if m and m.group("tin") else None,
               "tokens_out": int(m.group("tout")) if m and m.group("tout") else None,
               "annotation_seconds": p.annotation_seconds, "model_summary": p.model_summary,
               "warnings": p.warnings, "adapter_detail": p.adapter_detail,
               "annotations": [{"start": a.start, "end": a.end, "action": a.action, "confidence": a.confidence,
                                "objects": a.objects, "visual_evidence": a.visual_evidence,
                                "cited_frame_times": a.cited_frame_times,
                                "boundary_precision_s": a.boundary_precision_s} for a in sess.annotations],
               "qc_flags": [{"code": f.code, "severity": f.severity.value, "origin": f.origin.value,
                             "start": f.start, "end": f.end, "reason": f.reason} for f in sess.flags if not f.resolved]}
    except AdapterUnavailable as e:
        rec = {"status": "failed", "error": f"AdapterUnavailable: {e}"}
    except Exception as e:  # noqa: BLE001 - any failure must be recorded, never treated as an empty prediction
        rec = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
    rec["elapsed_s"] = round(time.perf_counter() - t0, 2)
    rec["called_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return {**rec, "from_cache": False}


# ------------------------------------------------------------------------------- report
def aggregate(episodes: list[dict]) -> dict:
    ok = [e for e in episodes if e.get("prediction", {}).get("status") == "ok" and e.get("metrics")]

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return {"mean": round(float(np.mean(vals)), 3) if vals else None, "n": len(vals)}

    timing = {k: mean([e["metrics"]["timing"][k]["abs_error_s"] for e in ok])
              for k in ("grasp", "release", "motion_start", "motion_end")}
    within = {}
    for tol in (0.5, 1.0):
        for k in ("grasp", "release"):
            errs = [e["metrics"]["timing"][k]["abs_error_s"] for e in ok if e["metrics"]["timing"][k]["measurable"]]
            within[f"{k}_within_{tol}s"] = {"count": sum(x <= tol for x in errs), "of": len(errs)}
    per_phase_iou = {p: mean([e["metrics"]["per_phase"][p]["iou"] for e in ok if p in e["metrics"]["per_phase"]])
                     for p in PHASES}
    return {"episodes_requested": len(episodes), "episodes_evaluated": len(ok),
            "denominator_note": "Aggregates use only episodes whose Muse call succeeded and whose reference was "
                                "derived; failed / skipped calls are excluded and listed separately.",
            "phase_coverage": mean([e["metrics"]["phase_coverage"] for e in ok]),
            "frame_agreement": mean([e["metrics"]["frame_agreement"] for e in ok]),
            "order_agreement": mean([e["metrics"]["order_agreement"] for e in ok]),
            "order_fully_correct": {"count": sum(bool(e["metrics"]["order_fully_correct"]) for e in ok), "of": len(ok)},
            "abs_timing_error_s": timing, "timing_hit_rates": within, "per_phase_iou": per_phase_iou,
            "unmapped_predictions_total": sum(len(e["metrics"]["unmapped_predictions"]) for e in ok),
            "tokens_in_total": sum(e["prediction"].get("tokens_in") or 0 for e in ok),
            "tokens_out_total": sum(e["prediction"].get("tokens_out") or 0 for e in ok),
            "live_calls_this_run": sum(not e["prediction"].get("from_cache", True) for e in episodes)}


LIMITATIONS = [
    "References are DERIVED from recorded joint/gripper signals by fixed rules (see rules), not human-annotated "
    "action boundaries; they inherit the thresholds' assumptions.",
    "Lift, transport and lowering are merged into 'transport', and placement + release into 'release', because "
    "the dataset records no end-effector height or object pose.",
    "Predicted labels are mapped to phases by fixed keyword rules; ambiguous wording (e.g. 'pick up') maps to a "
    "single phase and some labels may be unmapped.",
    "Muse Spark sees ~1 frame per second (RoboTrace's default sampling), so boundary errors below ~0.5 s are not "
    "meaningful.",
    "10 episodes of one task, one object, one camera and one robot: a proof of concept, not a benchmark.",
    "Confidence values are model self-reported and uncalibrated.",
]


def markdown_report(report: dict) -> str:
    agg = report["aggregate"]
    lines = [f"# RoboTrace × Muse Spark — {REPO} evaluation", "",
             f"Camera `{report['camera']}` · episodes {report['episodes_requested']} · dataset licence "
             f"`{report['license']}` · mapping `{report.get('mapping_version', 'v1')}` · prompt "
             f"`{report.get('prompt_version', LEGACY_PROMPT_VERSION)}` · generated {report['generated_at']}", "",
             "**Reference phases are rule-derived from recorded robot signals, not human annotations.**", "",
             "| Ep | Status | #pred | Coverage | Frame agr. | Order | Grasp err (s) | Release err (s) | "
             "Start err (s) | End err (s) | Unmapped | Open QC flags | Tokens in/out | Time (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def f(v):
        return "—" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))

    for e in report["episodes"]:
        p, m = e.get("prediction", {}), e.get("metrics")
        if m:
            t = m["timing"]
            lines.append(f"| {e['episode']} | {p['status']} | {len(p['annotations'])} | {f(m['phase_coverage'])} | "
                         f"{f(m['frame_agreement'])} | {f(m['order_agreement'])} | {f(t['grasp']['abs_error_s'])} | "
                         f"{f(t['release']['abs_error_s'])} | {f(t['motion_start']['abs_error_s'])} | "
                         f"{f(t['motion_end']['abs_error_s'])} | {len(m['unmapped_predictions'])} | "
                         f"{len(p.get('qc_flags', []))} | {f(p.get('tokens_in'))}/{f(p.get('tokens_out'))} | "
                         f"{f(p.get('annotation_seconds'))} |")
        else:
            lines.append(f"| {e['episode']} | {p.get('status', e.get('status'))} | — | — | — | — | — | — | — | — | — "
                         f"| — | — | — |")
    lines += ["", f"## Aggregate (n = {agg['episodes_evaluated']} of {agg['episodes_requested']} episodes)", "",
              agg["denominator_note"], "",
              f"- Phase coverage: {f(agg['phase_coverage']['mean'])} (n={agg['phase_coverage']['n']})",
              f"- Frame agreement: {f(agg['frame_agreement']['mean'])} (n={agg['frame_agreement']['n']})",
              f"- Order agreement: {f(agg['order_agreement']['mean'])}; fully correct order in "
              f"{agg['order_fully_correct']['count']}/{agg['order_fully_correct']['of']}"]
    for k, v in agg["abs_timing_error_s"].items():
        lines.append(f"- Mean |{k} error|: {f(v['mean'])} s (n={v['n']})")
    for k, v in agg["timing_hit_rates"].items():
        lines.append(f"- {k.replace('_', ' ')}: {v['count']}/{v['of']}")
    lines.append("- Per-phase IoU: " + ", ".join(f"{p} {f(v['mean'])} (n={v['n']})"
                                                  for p, v in agg["per_phase_iou"].items()))
    lines += ["", "## Failed calls / episodes needing manual inspection", ""]
    lines += [f"- ep {x['episode']}: {x['reason']}" for x in report["needs_attention"]] or ["- none"]
    lines += ["", "## Reference rules", "", "```json", json.dumps(REF_RULES, indent=2), "```", "",
              "## Phase-mapping rules (first match wins)", ""]
    lines += [f"- `{p}`: `{pat}`" for p, pat in PHASE_RULES]
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in LIMITATIONS]
    return "\n".join(lines) + "\n"


def needs_attention(episodes: list[dict]) -> list[dict]:
    out = []
    for e in episodes:
        p = e.get("prediction", {})
        if p.get("status") != "ok":
            out.append({"episode": e["episode"], "reason": f"{p.get('status', e.get('status'))}: "
                                                          f"{p.get('error') or e.get('error', '')}"})
            continue
        m = e.get("metrics") or {}
        if not p.get("annotations"):
            out.append({"episode": e["episode"], "reason": "call succeeded but returned 0 annotations"})
        if m.get("unmapped_predictions"):
            out.append({"episode": e["episode"], "reason": f"unmapped labels: {m['unmapped_predictions']}"})
        if e["reference"]["notes"]:
            out.append({"episode": e["episode"], "reason": "reference: " + "; ".join(e["reference"]["notes"])})
    return out


def report_stem(out: Path, name: str, overwrite: bool) -> str:
    """Never replace an existing report unless asked: fall back to a timestamped name."""
    if overwrite or not ((out / f"{name}.json").exists() or (out / f"{name}.md").exists()):
        return name
    return f"{name}_{time.strftime('%Y%m%d-%H%M%S')}"


# ------------------------------------------------------------------------------- main
def parse_episodes(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate Muse Spark annotations on lerobot/svla_so101_pickplace.")
    ap.add_argument("--episodes", default="0-9")
    ap.add_argument("--camera", default="side", choices=["side", "up"])
    ap.add_argument("--live", action="store_true", help="make live Muse Spark calls (otherwise cache only)")
    ap.add_argument("--max-calls", type=int, default=10)
    ap.add_argument("--retry-failed", action="store_true", help="re-call episodes whose cached call failed")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--prompt-version", default=None,
                    help="score cached predictions made with this prompt version (default: current). "
                         "Live calls are only allowed with the current version.")
    ap.add_argument("--report-name", default="report", help="report file stem (report -> report.json/.md)")
    ap.add_argument("--overwrite-report", action="store_true", help="allow replacing an existing report")
    args = ap.parse_args(argv)
    from robotrace.annotation_engine import OBSERVATION_PROMPT_VERSION
    from robotrace.config import get_settings as _gs
    from robotrace.sampling import sampling_config
    prompt_version = args.prompt_version or OBSERVATION_PROMPT_VERSION
    sampling = LEGACY_SAMPLING if prompt_version == LEGACY_PROMPT_VERSION else sampling_config(_gs())
    if args.live and prompt_version != OBSERVATION_PROMPT_VERSION:
        print(f"--live needs the current prompt version {OBSERVATION_PROMPT_VERSION}", file=sys.stderr)
        return 2
    cache = args.out / "cache"

    lic = dataset_license(cache)
    print(f"Dataset {REPO}: licence {lic}")
    if lic == "unknown":
        print("Licence could not be determined; stopping before downloading data.", file=sys.stderr)
        return 2
    info, tasks, eps_meta, data = load_metadata(cache)
    if info.get("fps") != FPS:
        print(f"Unexpected fps {info.get('fps')}", file=sys.stderr)
        return 2
    task_text = {int(r["task_index"]): r["index"] for r in tasks.to_dict("records")}
    table = video_index(cache, args.camera)
    key = f"videos/observation.images.{args.camera}"
    budget = {"calls": 0, "max_calls": args.max_calls}
    if args.live:
        from robotrace.config import get_settings
        if not get_settings().has_opencode_go_key:
            print("OPENCODE_GO_API_KEY is not configured; refusing to run live.", file=sys.stderr)
            return 2

    episodes = []
    for ep in parse_episodes(args.episodes):
        row = eps_meta[eps_meta.episode_index == ep]
        if row.empty:
            episodes.append({"episode": ep, "status": "missing", "error": "episode not in metadata"})
            continue
        row = row.iloc[0]
        d = data[data.episode_index == ep].sort_values("frame_index")
        entry = {"episode": ep, "task": task_text.get(int(d.task_index.iloc[0]), "?"), "frames": int(row["length"])}
        try:
            clip = prepare_clip(cache, args.camera, table, ep, float(row[f"{key}/from_timestamp"]),
                                float(row[f"{key}/to_timestamp"]), int(row["length"]))
        except Exception as e:  # noqa: BLE001
            entry.update(status="prep_failed", error=f"{type(e).__name__}: {e}", prediction={"status": "prep_failed",
                                                                                           "error": str(e)})
            episodes.append(entry)
            continue
        entry["clip"] = str(clip.relative_to(ROOT)) if clip.is_relative_to(ROOT) else str(clip)
        entry["reference"] = derive_reference(d.timestamp.to_numpy(np.float64),
                                              np.stack(d["observation.state"].to_numpy()).astype(np.float64),
                                              np.stack(d["action"].to_numpy()).astype(np.float64))
        model = _gs().muse_model
        pred = run_inference(clip, predictions_dir(args.out, prompt_version, sampling, model)
                             / f"ep{ep:03d}_{args.camera}.json", live=args.live, retry_failed=args.retry_failed,
                             budget=budget, prompt_version=prompt_version, sampling=sampling, model=model)
        entry["prediction"] = pred
        if pred["status"] == "ok":
            entry["metrics"] = score_episode(entry["reference"], pred["annotations"])
        print(f"ep {ep}: clip ready, reference events {entry['reference']['events']}, prediction {pred['status']}"
              f"{' (cached)' if pred.get('from_cache') else ''}")
        episodes.append(entry)

    report = {"dataset": REPO, "license": lic, "camera": args.camera, "fps": FPS,
              "mapping_version": MAPPING_VERSION, "prompt_version": prompt_version, "sampling": sampling,
              "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "episodes_requested": len(episodes),
              "reference_rules": REF_RULES, "phase_rules": PHASE_RULES, "episodes": episodes,
              "aggregate": aggregate(episodes), "needs_attention": needs_attention(episodes),
              "limitations": LIMITATIONS}
    args.out.mkdir(parents=True, exist_ok=True)
    stem = report_stem(args.out, args.report_name, args.overwrite_report)
    (args.out / f"{stem}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (args.out / f"{stem}.md").write_text(markdown_report(report), encoding="utf-8")
    print(f"\nLive calls this run: {budget['calls']}. Report: {args.out / (stem + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
