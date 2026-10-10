"""RoboTrace remote inference service for a Colab / Kaggle GPU notebook.

Implements the contract in docs/REMOTE_API.md:
  GET  /health   -> {"status": "ok", "model": "<id>"}
  POST /annotate (multipart file=<video>, header Authorization: Bearer <token>)
       -> {"model": "<id>", "segments": [{"start", "end", "action", "confidence": null}]}

Model: Qwen/Qwen2-VL-2B-Instruct (Apache-2.0) by default; override with ROBOTRACE_VLM_MODEL
(e.g. Qwen/Qwen2.5-VL-7B-Instruct, Apache-2.0, needs a larger GPU).
Frames are sampled at ~1 fps (max ROBOTRACE_VLM_FRAMES) and passed as timestamped images, the same
frame-based approximation the Claude adapter uses. Timestamps are therefore quantised to ~1s.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
from pathlib import Path

import cv2
import torch
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL_ID = os.environ.get("ROBOTRACE_VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
MAX_FRAMES = int(os.environ.get("ROBOTRACE_VLM_FRAMES", "24"))
MAX_MB = float(os.environ.get("ROBOTRACE_MAX_UPLOAD_MB", "200"))
TOKEN = os.environ.get("ROBOTRACE_REMOTE_TOKEN") or secrets.token_urlsafe(24)

app = FastAPI(title="RoboTrace remote inference")
_model = None
_processor = None


def load_model():
    global _model, _processor
    if _model is None:
        if "2.5" in MODEL_ID:
            from transformers import Qwen2_5_VLForConditionalGeneration as cls
        else:
            cls = Qwen2VLForConditionalGeneration
        _model = cls.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
        _processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=128 * 28 * 28, max_pixels=320 * 28 * 28)
    return _model, _processor


def check_token(authorization: str = Header(default="")):
    if not secrets.compare_digest(authorization, f"Bearer {TOKEN}"):
        raise HTTPException(status_code=401, detail="invalid token")


def sample_frames(path: Path) -> tuple[list[tuple[float, Image.Image]], float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or n <= 0:
        raise HTTPException(status_code=422, detail="unreadable video")
    dur = n / fps
    k = int(min(MAX_FRAMES, max(4, round(dur))))
    out = []
    for i in range(k):
        t = (i + 0.5) * dur / k
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if ok:
            out.append((round(t, 2), Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
    cap.release()
    return out, dur


PROMPT = ("These are frames from a {dur:.1f}s robot manipulation video, each preceded by its timestamp. "
          "List the robot's actions in order as JSON only, no prose: "
          '[{{"start": seconds, "end": seconds, "action": "short imperative description naming the object"}}]. '
          "Segments must not overlap. Do not invent actions while the robot is idle.")


def parse_segments(text: str, dur: float) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    segs = []
    for s in raw if isinstance(raw, list) else []:
        try:
            st, en, act = float(s["start"]), float(s["end"]), str(s["action"]).strip()[:200]
        except (KeyError, TypeError, ValueError):
            continue
        if act:
            segs.append({"start": st, "end": en, "action": act, "confidence": None})
    return segs  # returned as-is (even if invalid) so RoboTrace QC can flag model timestamp errors


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "gpu": torch.cuda.is_available()}


@app.post("/annotate", dependencies=[Depends(check_token)])
async def annotate(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="file too large")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "video.mp4"
        p.write_bytes(data)
        frames, dur = sample_frames(p)
    model, processor = load_model()
    content = [{"type": "text", "text": PROMPT.format(dur=dur)}]
    for t, _img in frames:
        content += [{"type": "text", "text": f"t={t:.1f}s"}, {"type": "image"}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img for _t, img in frames], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=768, do_sample=False)
    reply = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return {"model": MODEL_ID, "segments": parse_segments(reply, dur), "frames_used": len(frames)}
