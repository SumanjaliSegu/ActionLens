"""Video decoding, validation, frame sampling, and deterministic pixel signals.

Everything here is model-free. Timestamps are seconds; frame indices are 0-based.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from .config import Settings
from .schemas import EvidenceFrame, VideoMetadata

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class VideoError(ValueError):
    """Raised for user-facing video problems (bad format, too long, corrupt...)."""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def probe(path: Path, display_name: Optional[str] = None) -> VideoMetadata:
    """Read container metadata and verify at least one frame decodes."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError("Could not open the video. The file may be corrupt or use an unsupported codec.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, _ = cap.read()
        if not ok:
            raise VideoError("The video contains no decodable frames.")
    finally:
        cap.release()
    if fps <= 0 or fps > 1000:
        raise VideoError(f"Video reports an invalid frame rate ({fps}).")
    if n <= 0:
        raise VideoError("Video reports zero frames.")
    return VideoMetadata(
        filename=display_name or path.name,
        sha256=sha256_file(path),
        duration_s=round(n / fps, 3),
        fps=round(fps, 3),
        frame_count=n,
        width=w,
        height=h,
        size_bytes=path.stat().st_size,
    )


def validate_upload(filename: str, size_bytes: int, settings: Settings) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise VideoError(f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}.")
    if size_bytes > settings.max_upload_mb * 1024 * 1024:
        raise VideoError(f"File is {size_bytes / 1e6:.1f} MB; the limit is {settings.max_upload_mb:.0f} MB.")
    if size_bytes == 0:
        raise VideoError("The uploaded file is empty.")
    validate_disk_space(size_bytes, settings)


def validate_disk_space(size_bytes: int, settings: Settings, safety_factor: float = 3.0) -> None:
    """The upload itself plus a transcoded copy and extracted frames can together need several times the
    original file size on disk. Refuse early with a clear message rather than failing partway through."""
    try:
        settings.work_dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(settings.work_dir).free
    except OSError:
        return  # cannot determine free space; do not block the upload on that basis
    needed = size_bytes * safety_factor
    if free < needed:
        raise VideoError(
            f"Not enough temporary disk space: {free / 1e6:.0f} MB free, ~{needed / 1e6:.0f} MB needed "
            f"(upload plus transcoding and frame cache). Free up space or reduce ROBOTRACE_MAX_UPLOAD_MB.")


def validate_duration(meta: VideoMetadata, settings: Settings) -> None:
    if meta.duration_s > settings.max_duration_s:
        raise VideoError(
            f"Video is {meta.duration_s:.0f}s long; the limit is {settings.max_duration_s:.0f}s "
            "(set ROBOTRACE_MAX_DURATION_S to change it).")


def time_to_frame(t: float, fps: float, frame_count: int) -> int:
    return int(min(max(round(t * fps), 0), max(frame_count - 1, 0)))


def frame_to_time(idx: int, fps: float) -> float:
    return idx / fps


def uniform_times(duration: float, n: int) -> list[float]:
    """n timestamps at the centres of n equal bins over [0, duration)."""
    if n <= 0 or duration <= 0:
        return []
    step = duration / n
    return [round((i + 0.5) * step, 3) for i in range(n)]


def extract_frames(path: Path, meta: VideoMetadata, times: Iterable[float], out_dir: Path,
                   max_width: int = 640) -> list[EvidenceFrame]:
    """Decode frames at the given timestamps (one sequential pass from the first wanted frame) and save
    JPEGs to out_dir.

    Seeks to the first wanted frame before scanning, instead of always starting at frame 0: without this,
    a call for frames late in a long video (e.g. one windowed-sampling window of several) must grab/decode
    through everything before it every time, which is O(video length) per call and compounds badly across
    many windows (measured: ~550s of pure decode time for 10 windows on a 768s video, before any model
    call). Seek landing position is backend/codec-dependent (it may land on the nearest prior keyframe,
    not exactly the requested frame), so the actual landed position is read back and used as the scan
    start rather than trusted blindly; the sequential grab/read loop below is unchanged, so frame-index
    correctness does not depend on seek accuracy, only speed does.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted: dict[int, float] = {}
    for t in times:
        wanted.setdefault(time_to_frame(t, meta.fps, meta.frame_count), t)
    if not wanted:
        return []
    results: list[EvidenceFrame] = []
    cap = cv2.VideoCapture(str(path))
    try:
        first, last = min(wanted), max(wanted)
        idx = 0
        if first > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            landed = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if 0 <= landed <= first:
                idx = landed
        while idx <= last:
            if idx in wanted:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = _resize(frame, max_width)
                fp = out_dir / f"f{idx:06d}.jpg"
                if not fp.exists():
                    cv2.imwrite(str(fp), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                results.append(EvidenceFrame(t=round(frame_to_time(idx, meta.fps), 3), frame_index=idx, path=str(fp)))
            elif not cap.grab():
                break
            idx += 1
    finally:
        cap.release()
    return results


def _resize(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


@dataclass
class SignalProfile:
    """Deterministic per-step pixel statistics.

    motion[i]: mean absolute grey-level difference (0-255) between the frame at times[i]
               and the frame one step earlier. A proxy for "something is moving".
    cut[i]:    1 - histogram correlation between the same two frames. Large values indicate
               an abrupt scene change (camera cut, occlusion, lighting jump).
    """
    times: np.ndarray
    motion: np.ndarray
    cut: np.ndarray
    step: float

    def window(self, start: float, end: float) -> np.ndarray:
        return (self.times >= start) & (self.times <= end)

    def mean_motion(self, start: float, end: float) -> float:
        m = self.window(start, end)
        return float(self.motion[m].mean()) if m.any() else 0.0

    @property
    def active_threshold(self) -> float:
        """Adaptive threshold separating 'active' from 'idle' steps for this video."""
        if self.motion.size == 0:
            return 0.0
        lo, hi = np.percentile(self.motion, 10), np.percentile(self.motion, 90)
        return float(max(lo + 0.25 * (hi - lo), 0.5))

    @property
    def subtle_threshold(self) -> float:
        """Above the noise floor but below `active_threshold`: small motions such as a gripper closing."""
        if self.motion.size == 0:
            return 0.0
        floor = float(np.percentile(self.motion, 10))
        return floor + 0.15 * max(self.active_threshold - floor, 0.0)

    def to_dict(self) -> dict:
        return {"times": self.times.tolist(), "motion": self.motion.tolist(),
                "cut": self.cut.tolist(), "step": self.step}


def compute_signals(path: Path, meta: VideoMetadata, hz: float = 5.0, width: int = 160) -> SignalProfile:
    """Single decoding pass computing motion energy and cut scores at `hz` samples per second."""
    stride = max(int(round(meta.fps / hz)), 1)
    times, motion, cut = [], [], []
    prev_gray, prev_hist = None, None
    cap = cv2.VideoCapture(str(path))
    try:
        idx = 0
        while True:
            if idx % stride == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                small = _resize(frame, width)
                gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
                # Luminance histogram: sensitive to cuts, occlusions and lighting jumps.
                hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
                cv2.normalize(hist, hist)
                if prev_gray is not None:
                    times.append(idx / meta.fps)
                    motion.append(float(np.mean(cv2.absdiff(gray, prev_gray))))
                    corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    cut.append(float(max(0.0, 1.0 - corr)))
                prev_gray, prev_hist = gray, hist
            elif not cap.grab():
                break
            idx += 1
    finally:
        cap.release()
    return SignalProfile(np.asarray(times, float), np.asarray(motion, float), np.asarray(cut, float),
                         stride / meta.fps)


def ffmpeg_exe() -> Optional[str]:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def make_browser_playable(src: Path, dst: Path, timeout_s: int = 300) -> Path:
    """Transcode to H.264/AAC MP4 so every browser can play it. Falls back to the source on failure.

    Arguments are passed as a list (no shell), so file names cannot inject commands.
    """
    exe = ffmpeg_exe()
    if exe is None:
        return src
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)]
    try:
        subprocess.run(cmd, check=True, timeout=timeout_s, capture_output=True)
        return dst
    except (subprocess.SubprocessError, OSError):
        return src
