# RoboTrace export schema — `robotrace.v1`

All exports are UTF-8. **All timestamps are seconds (float) from the start of the video.** Frame indices are
0-based indices into the decoded stream (`frame_index / fps ≈ t`).

## Common header (JSON exports)

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | str | `"robotrace.v1"` |
| `export_kind` | str | `annotations` or `review_package` |
| `exported_at` | ISO-8601 UTC | |
| `time_unit` | str | always `"seconds"` |
| `video` | object | `filename`, `sha256`, `duration_s`, `fps`, `frame_count`, `width`, `height`, `size_bytes`, `is_synthetic_sample` |
| `processing` | object | `adapter`, `adapter_detail`, `is_live_inference`, `is_precomputed_sample`, `annotation_seconds`, `qc_seconds`, `sampled_frames`, `warnings[]` |

## Annotation object

| Field | Type | Meaning |
|---|---|---|
| `id` | str | stable id (`ann_xxxxxxxx`) |
| `start`, `end` | float | seconds |
| `action` | str | action description |
| `evidence` | list | `{t, frame_index}` frames sampled at start / middle / end |
| `confidence` | float \| null | **not a calibrated probability**; see `confidence_kind` |
| `confidence_kind` | str | `none`, `model_self_reported`, `heuristic` |
| `source` | str | adapter name; `+edited` suffix after reviewer edits; `human` for reviewer-added |
| `review_status` | str | `unreviewed`, `correct`, `incorrect`, `needs_review` (reviewed exports only) |
| `reviewer_note` | str | free text |

## Flag object (review package)

| Field | Meaning |
|---|---|
| `code` | e.g. `TS_OVERLAP` (see table below) |
| `category` | `timestamp`, `evidence`, `boundary`, `missing_candidate` |
| `severity` | `high`, `medium`, `low` |
| `origin` | `deterministic_rule`, `visual_signal`, `model_judgment` |
| `annotation_id` | target annotation, or null for interval findings |
| `start`, `end` | seconds |
| `reason` | human-readable explanation |
| `details` | raw numbers behind the reason |
| `resolved`, `resolution_note` | reviewer resolution |

### Flag codes

| Code | Origin | Severity | Fires when |
|---|---|---|---|
| `TS_NEGATIVE` | rule | high | start or end < 0 |
| `TS_REVERSED` | rule | high | end < start |
| `TS_ZERO_LENGTH` | rule | medium | end == start |
| `TS_TOO_SHORT` | rule | low | duration < 0.25 s |
| `TS_OUT_OF_RANGE` | rule | high | timestamp > video duration |
| `TS_OVERLAP` | rule | medium | two valid intervals overlap by > 0.05 s (both are flagged) |
| `TS_LONG_SEGMENT` | rule | low | duration > 4 s and > 2× median duration |
| `TS_GAP` | rule | low | ≥ 1 s of video not covered by any annotation |
| `TXT_EMPTY` | rule | medium | empty description |
| `EV_LOW_MOTION` | signal | medium / low | < 20 % of the segment shows clear motion; low if subtle motion is present |
| `EV_IDLE_INSIDE` | signal | medium | ≥ 1 s motionless stretch covering ≥ 25 % of an otherwise active segment |
| `BD_START_MAY_BE_LATE` | signal | medium | motion in the 0.6 s before start, uncovered by other annotations |
| `BD_END_MAY_BE_EARLY` | signal | medium | motion in the 0.6 s after end, uncovered by other annotations |
| `BD_ABRUPT_CHANGE` | signal | low | luminance-histogram change spike not within 0.4 s of a boundary |
| `MISSING_CANDIDATE` | signal | medium / low | uncovered interval ≥ 0.5 s with ≥ 50 % clear (medium) or subtle (low) motion |
| `MODEL_CONTRADICTED` | model | high | vision model says frames contradict the description |
| `MODEL_INSUFFICIENT` | model | low | vision model cannot judge from the frames |

Review priority = 3 × high + 2 × medium + 1 × low **unresolved** flags. It orders the queue and is not a probability.

## Review package (`export_kind = review_package`)

`baseline_annotations` (engine output, never mutated), `reviewed_annotations`, `deleted_annotation_ids`,
`initial_flags` (QC before review), `current_flags` (QC after edits, with resolutions), `model_verdicts`,
`review_events` (`{at, kind, target_id, payload}` for every status change, edit, add, delete, flag resolution,
re-check), and `summary`.

## CSV

Columns: `id, start_s, end_s, duration_s, action, source, confidence, confidence_kind, review_status,
reviewer_note, open_flags, flag_codes, evidence_times_s`. Text that begins with `= + - @` is prefixed with `'`
to prevent spreadsheet formula injection.

## Reference annotations (for evaluation upload)

```json
{"segments": [{"start": 1.5, "end": 4.5, "action": "reach toward the red cube"}],
 "erroneous_ids": ["ann_s00"],
 "missing": [{"start": 19.0, "end": 20.0, "action": "grasp the blue cube"}]}
```
Only `segments` is required. `erroneous_ids` and `missing` enable flag precision/recall.
