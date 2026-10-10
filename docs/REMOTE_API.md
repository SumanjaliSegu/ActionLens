# Remote inference contract

The web app's `remote` adapter talks to any service that implements these two endpoints. The reference
implementation is `remote_inference/server.py`, launched from `notebooks/robotrace_remote_gpu.ipynb`.

## `GET /health`

Response `200`:
```json
{"status": "ok", "model": "Qwen/Qwen2-VL-2B-Instruct", "gpu": true}
```
The app calls this with a 5 s timeout before every request and shows the result under **Engine status**.
Any error, timeout, or non-`ok` status marks the engine unavailable; the rest of the app keeps working.

## `POST /annotate`

- Header: `Authorization: Bearer <ROBOTRACE_REMOTE_TOKEN>`
- Body: `multipart/form-data` with field `file` (the video)
- Timeout on the client: `ROBOTRACE_REMOTE_TIMEOUT_S` (default 300 s)

Response `200`:
```json
{"model": "Qwen/Qwen2-VL-2B-Instruct",
 "segments": [{"start": 1.0, "end": 4.0, "action": "reach toward the red cube", "confidence": null}],
 "frames_used": 24}
```
Times are seconds. Segments are returned as the model produced them, even if they overlap or run out of range,
so that RoboTrace QC can flag those errors instead of having them hidden.

Errors: `401` bad token, `413` file too large, `422` unreadable video. The client converts every failure to a
readable message and never substitutes canned annotations.

## Session lifetime

Colab/Kaggle sessions are temporary (idle timeouts, ~12 h maximum). The cloudflared quick-tunnel URL changes
every time the notebook restarts, so update the `ROBOTRACE_REMOTE_URL` secret after each restart. For a
persistent endpoint, run `server.py` on a paid GPU host (e.g. an HF Inference Endpoint or a GPU VM); the
contract is unchanged.
