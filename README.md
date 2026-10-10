---
title: RoboTrace
emoji: 🎯
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# RoboTrace

**Evidence-grounded robotics video annotation & quality control.**

Robot-learning datasets need action-level annotations: what the robot did, and when. Automatic annotators make
mistakes: wrong labels, sloppy boundaries, invented actions, missed actions. Checking every label by hand is
slow. RoboTrace is a **quality-control layer**. It generates or accepts timestamped action annotations,
checks each one against the video's own evidence, and gives a human reviewer a ranked queue of suspicious items,
each with the reason it was flagged. Review decisions are recorded and exported.

> RoboTrace does not claim to have invented automatic annotation or to beat existing systems. Its flags are
> prompts for human review, not verdicts.

- [Architecture](docs/ARCHITECTURE.md) · [Export schema & flag codes](docs/SCHEMA.md) ·
  [Remote GPU API](docs/REMOTE_API.md) · [5-minute demo script](docs/DEMO_SCRIPT.md)

---

## What works

| Area | Status |
|---|---|
| Landing page, pipeline diagram, honest-notes panel | ✅ |
| Upload (mp4/mov/avi/mkv/webm/m4v), size & duration limits, corrupt-file handling, H.264 transcode for playback, temp cleanup | ✅ |
| Bundled **synthetic** sample + precomputed **simulated** annotations (clearly labelled) | ✅ |
| Annotation adapters: `sample`, `motion` (CPU), `muse` / `gemini` / `claude` (vision APIs), `remote` (GPU notebook) | ✅ code paths tested offline with mocked APIs; live runs need credentials (see below) |
| QC layer 1 — deterministic timestamp rules | ✅ |
| QC layer 2 — pixel-signal evidence (motion energy, luminance cuts), boundary checks, missing-action candidates | ✅ |
| QC layer 3 — model verifier (Claude vision judges start/mid/end frames) | ✅ needs `ANTHROPIC_API_KEY` |
| Transparent review priority | ✅ |
| Review UI: player with seek, timeline, evidence frames, flags with reasons and raw values, correct / incorrect / needs-review, edit text & timestamps, add from candidate, delete, resolve / reopen flags, re-run QC, reset | ✅ |
| Dashboard: counts, flag categories before/after, status, sortable/filterable table, before→after diff | ✅ |
| Evaluation vs reference (IoU P/R, boundary error, label accuracy, flag P/R) | ✅ automatic for the sample; upload a reference JSON for other videos |
| Exports: annotations JSON, CSV, Markdown QC report, full review package | ✅ |
| Colab/Kaggle GPU notebook + FastAPI service | ✅ written; **not executed in this environment (no GPU here)** |

## Quick start (local)

Requires Python 3.10–3.12.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app.py              # http://localhost:8501
pytest -q                         # run the tests
```

Click **Try the sample**. No GPU, API key, or download is needed. To regenerate the sample:
`python scripts/make_sample.py` (deterministic rendering; ffmpeg is supplied by `imageio-ffmpeg`).

## Annotation engines

| Engine | Live? | Needs | Output |
|---|---|---|---|
| `sample` | **No**: precomputed | nothing | Simulated annotator output for the bundled sample **only** (SHA-256 checked; refused for any other video) |
| `motion` | yes | nothing (CPU) | Timing of motion bursts, labelled `activity segment N (unlabeled motion)` |
| `claude` | yes | `ANTHROPIC_API_KEY` | Labelled segments from ≤ 24 timestamped frames (~1 fps) |
| `muse` (primary) | yes | `OPENCODE_GO_API_KEY` | Muse Spark via OpenCode Go (`/responses`). Same frame sampling and observation fields as `gemini`. Contributor models may use the frames you send to train Meta models (not zero-data-retention) |
| `gemini` | yes | `GEMINI_API_KEY` | Same frame sampling; also returns objects, visual evidence, cited frames and a scene summary. Boundaries are marked approximate (± half the frame spacing) |
| `remote` | yes | `ROBOTRACE_REMOTE_URL` (+ token) | Labelled segments from the GPU notebook (Qwen2-VL-2B by default) |

The `muse`, `gemini`, `claude` and `remote` engines are **frame-based approximations**: they see stills, so boundaries are quantised
to the sampling interval and very short actions can be missed. Confidence values are model self-reports and
are labelled as uncalibrated. If an engine is unavailable, the UI says why and the others keep working.
Uploaded videos never receive canned annotations.

## Quality control

Every flag carries an **origin** and a human-readable **reason** plus the raw numbers behind it:

1. **Deterministic rules** (`deterministic_rule`): negative, reversed, zero-length, too-short, out-of-range,
   and overlapping intervals; segments far longer than the median; uncovered gaps. A gap alone is never called a
   missing action.
2. **Pixel signals** (`visual_signal`): frame-difference motion energy at 5 Hz with an adaptive threshold and
   noise floor, plus a luminance-histogram change score.
   - Actions with no visible change (MEDIUM), or with only subtle change (LOW, because gripper-only steps are near the signal's sensitivity limit).
   - Idle stretches inside segments.
   - Starts that come after motion is already under way, and ends that come before motion stops.
   - Abrupt changes that no boundary explains.
   - Uncovered motion, reported as **possible missing action (candidate)**.
3. **Model judgment** (`model_judgment`, optional): a vision model sees each segment's start/middle/end frames
   and returns supported / contradicted / insufficient. Verdicts are discarded automatically once the reviewer
   edits that annotation.

Priority = 3 × high + 2 × medium + 1 × low unresolved flags. This is a triage ordering, **not a probability**.
QC never edits data. Edits re-run QC, and resolutions persist across re-checks.

## Results on the bundled sample (measured, synthetic only)

The sample is a rendered 25 s pick-and-place clip (`scripts/make_sample.py`). The renderer's script gives exact
ground truth. The "baseline annotations" are **simulated**: ground truth with six injected annotation errors
(hallucinated action, late start, wrong label, overlong segment, reversed timestamps, out-of-range end) and one
omitted action. Initial QC (rules + signals, no model) on it:

- All 6 erroneous annotations are flagged, plus 2 correct ones (an overlap partner, and a grasp with subtle
  motion): annotation-level flag precision 0.75 and recall 1.0.
- The omitted "grasp the blue cube" interval gets a gap flag and a low-severity missing-action candidate.
- **The wrong label is flagged only incidentally** (by a low-motion flag). Rules cannot detect wrong labels;
  that needs the model verifier.

These numbers come from a synthetic clip with errors we designed. **They say nothing about performance on real
robot footage.** No real-data benchmark has been run.

## Deploying to a public URL

The app is CPU-only and stateless, so any free Streamlit host works. GPU inference stays optional and remote.

### Option A: Hugging Face Spaces (Docker) — recommended

1. Create a Space → SDK **Docker** → blank template.
2. Push this repository to the Space (`git remote add space https://huggingface.co/spaces/<user>/robotrace`,
   then `git push space main`). The YAML header of this README configures the Space (port 7860), and the
   `Dockerfile` builds it.
3. *Settings → Variables and secrets*: add `ANTHROPIC_API_KEY` (optional), plus `ROBOTRACE_REMOTE_URL` and
   `ROBOTRACE_REMOTE_TOKEN` (optional).
4. Open `https://<user>-robotrace.hf.space` and click **Try the sample**.

Free CPU Spaces sleep after inactivity, and the first load after sleep takes about 30–60 s. Open it before presenting.

### Option B: Streamlit Community Cloud

1. Push to a GitHub repository.
2. On share.streamlit.io, create a new app with main file `app.py` and Python 3.11.
3. Under *Advanced settings → Secrets*, add (TOML):
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   Streamlit exposes top-level secrets as environment variables, which is how RoboTrace reads them.

### Optional: GPU inference via Colab/Kaggle

1. Open `notebooks/robotrace_remote_gpu.ipynb` in Colab or Kaggle with a GPU runtime.
2. Upload `remote_inference/server.py` (or set `REPO_RAW` to its raw URL), then run all cells.
3. Copy the printed `ROBOTRACE_REMOTE_URL` / `ROBOTRACE_REMOTE_TOKEN` into the host's secrets and restart the app.
4. The sidebar's **Engine status** shows whether the endpoint is healthy. When the notebook stops, the engine
   shows as unavailable and everything else keeps working. The quick-tunnel URL changes on every restart.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Enables the `claude` engine and the model verifier (secret) |
| `ROBOTRACE_CLAUDE_MODEL` | `claude-opus-5-5` | Claude model id |
| `OPENCODE_GO_API_KEY` | — | Enables the `muse` engine (secret; sent only as `Authorization: Bearer`) |
| `ROBOTRACE_MUSE_MODEL` | `muse-spark-1.3-contributor` | OpenCode Go model id |
| `ROBOTRACE_OPENCODE_GO_URL` | `https://opencode.ai/zen/go/v1` | OpenCode Go base URL |
| `ROBOTRACE_OPENCODE_TIMEOUT_S` | `180` | OpenCode Go request timeout |
| `ROBOTRACE_SAMPLING` | `legacy` | Frame sampling for `muse` / `gemini`: `legacy` = one request, ~1 frame/s, ≤ `ROBOTRACE_MAX_FRAMES` frames (spacing grows with video length); `windowed` = fixed spacing over overlapping windows, one request per window (more requests and tokens; see `robotrace/sampling.py`) |
| `ROBOTRACE_FRAME_SPACING_S` | `0.5` | Target frame spacing in `windowed` mode |
| `ROBOTRACE_WINDOW_OVERLAP_S` | `1.0` | Overlap between consecutive windows |
| `ROBOTRACE_MAX_WINDOWS` | `4` | Request cap per video in `windowed` mode; beyond it the spacing is widened (with a warning) |
| `GEMINI_API_KEY` | — | Enables the `gemini` engine (secret; sent only in the `x-goog-api-key` header) |
| `ROBOTRACE_GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model id |
| `ROBOTRACE_GEMINI_TIMEOUT_S` | `180` | Gemini request timeout |
| `ROBOTRACE_REMOTE_URL` | — | Base URL of the GPU notebook service |
| `ROBOTRACE_REMOTE_TOKEN` | — | Bearer token for that service (secret) |
| `ROBOTRACE_REMOTE_TIMEOUT_S` | 300 | Remote request timeout |
| `ROBOTRACE_MAX_UPLOAD_MB` | 200 | Upload size limit (keep `.streamlit/config.toml` `maxUploadSize` ≥ this) |
| `ROBOTRACE_MAX_DURATION_S` | 180 | Video duration limit |
| `ROBOTRACE_MAX_FRAMES` | 24 | Frames per vision-model request (`muse`, `gemini`, `claude`; cost/latency bound) |
| `ROBOTRACE_WORK_DIR` | `<tmp>/robotrace` | Uploads and frame cache. Uploads older than 6 h are deleted |

See `.env.example`. Never commit real keys. `.env` and `.streamlit/secrets.toml` are git-ignored (and excluded
from Docker builds). A `.env` file in the project root is loaded automatically at startup; variables already set
in the shell or host secrets take precedence, and `ROBOTRACE_SKIP_DOTENV=1` disables loading (the tests do this).
`ANTHROPIC_AUTH_TOKEN` is accepted as an alternative to `ANTHROPIC_API_KEY`.

**Cost note:** the `muse`, `gemini` and `claude` engines and the verifier are paid API calls (up to
`ROBOTRACE_MAX_FRAMES` images per request; `ROBOTRACE_SAMPLING=windowed` makes up to `ROBOTRACE_MAX_WINDOWS`
requests per video, and the Analyze page shows the planned count before you run). The sample, `motion`, QC,
review and export features are free.

## Security & reliability

- Uploads: extension allow-list, size and duration limits, decode check. Files are stored under random names
  (the user's filename is never used as a path), and the previous upload is deleted on a new upload.
- ffmpeg is invoked with an argument list (no shell). Exports drop server file paths, and CSV text is
  neutralised against formula injection.
- Model and remote output are schema-validated (Pydantic). Invented ids are ignored, and failures become readable
  errors, never canned data.
- Reference uploads are parsed as JSON only, with no pickle or other unsafe deserialisation.
- Reviewer and model text is escaped before rendering (HTML and Markdown syntax, so model output cannot inject
  links or images that make the browser fetch external URLs).
- Frame-based model output is accepted only if its timestamps match frames that were actually sent; failures in
  any request fail the whole run (no partial results, no automatic retries).

## Tests

`pytest -q` covers:
- timestamp rules, overlaps, gaps, and long segments
- signal flags: low motion, subtle motion, missing candidates, late starts, and abrupt changes
- stale model verdicts and priority ranking
- resolution carry-over
- review editing, validation, add/delete, and the event log
- every export format, path stripping, and CSV injection
- evaluation metrics
- video probing, corrupt-file rejection, upload validation, and frame extraction
- the motion adapter
- adapter unavailability, and an unreachable remote endpoint
- an end-to-end run on the sample
- Muse / Gemini / Claude adapters with **mocked** HTTP (request format, parsing, every error path, no fallback);
  these do not measure real model accuracy
- frame-sampling plans, cross-window merging and timestamp grounding (`robotrace/sampling.py`)
- the review → export workflow through the real Streamlit form, and escaping of untrusted model text
- the SO101 evaluator (reference rules, phase mapping, scoring, cache identity, never overwriting results)

On Windows without an activated venv: `.venv\Scripts\python.exe -m pytest -q`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Video won't play in the browser | The app transcodes uploads to H.264 via `imageio-ffmpeg`. If that failed, the original is shown. Convert to H.264 MP4 and re-upload. Evidence frames and QC still work. |
| "Could not open the video" | Corrupt file or unsupported codec. Re-encode: `ffmpeg -i in.mov -c:v libx264 out.mp4` |
| `claude` engine greyed out | `ANTHROPIC_API_KEY` is not set in the environment or host secrets. Restart the app after setting it. |
| `remote` unavailable | The notebook stopped, or the URL changed. Re-run the notebook and update the secret. |
| Upload rejected for size | Raise `ROBOTRACE_MAX_UPLOAD_MB` **and** `server.maxUploadSize` in `.streamlit/config.toml`. |
| Motion engine finds nothing | Very static or very noisy video. The threshold is adaptive per video. Annotate manually from Review. |
| Review state disappeared | State lives in the browser session. Download the review package before closing the tab. |

## Limitations & future work

- **Frame-based models.** The live engines see still frames, so boundaries are approximate (about 1 s) and brief
  actions can be missed. Future work: native video-LLMs with timestamp grounding, and per-boundary dense re-sampling.
- **Motion signals are global.** Fine manipulation (gripper only) produces weak signals, and camera motion inflates
  them. Future work: optical flow restricted to robot/object masks, and ego-motion compensation.
- **No automatic boundary refinement.** RoboTrace flags suspicious boundaries but never changes timestamps; only
  `BD_START_IN_IDLE` suggests a later start (`suggested_start_s`) for a reviewer to accept or reject.
- **No human-labelled real-data benchmark yet.** Besides the synthetic sample, `scripts/eval_so101.py` scores Muse
  on 10 `lerobot/svla_so101_pickplace` episodes against phases *derived by rules from recorded robot signals*
  (not human labels). Next: datasets with human action labels, licences checked.
- **Single-user, in-memory state.** There is no database or multi-reviewer agreement. The review package is the
  persistence mechanism.
- Model verifier verdicts are opinions. Measure their precision on a reference set before trusting them.

## Project layout

```
app.py                     Streamlit UI
robotrace/                 schemas, video_processor, annotation_engine, quality_engine,
                           review_service, export_service, evaluation, pipeline, charts, storage, config
data/sample/               synthetic sample video, ground truth, simulated baseline annotations
scripts/make_sample.py     regenerates the sample
scripts/run_inference.py   one video through the full pipeline from the command line (writes a review package)
scripts/eval_so101.py      SO101 evaluation (dry run by default; --live makes paid Muse calls, at most one per
                           episode, cached under out/eval_so101/ and never overwritten)
robotrace/sampling.py      legacy / windowed frame-sampling plans and cross-window merging
out/                       generated outputs and evaluation caches (git-ignored)
remote_inference/server.py FastAPI GPU service;  notebooks/robotrace_remote_gpu.ipynb launches it
tests/                     pytest suite
docs/                      architecture, schema, remote API, demo script
Dockerfile, .streamlit/    deployment
```

Sample video and annotations © the RoboTrace authors (generated by code in this repo), MIT licensed.
