# Five-minute demo script

**Setup before going on stage:** open the public URL and click through once (to warm the container). If an
`ANTHROPIC_API_KEY` secret is configured, pre-run "Run model verifier" once on the sample in a spare tab to
confirm the key works.

| Time | Screen | Say / do |
|---|---|---|
| 0:00–0:40 | **Overview** | "Robot learning needs action-level labels: what happened and when. Automatic annotators produce wrong labels, sloppy boundaries and missed actions, and checking everything by hand is slow. RoboTrace doesn't claim to annotate better. It is a **quality-control layer**: it checks every annotation against the video's own evidence and gives reviewers a ranked queue with reasons." Point at the pipeline diagram and the honest-notes box. |
| 0:40–1:00 | Click **Try the sample** | "This is a synthetic pick-and-place clip we rendered, so we know the exact truth. The annotations come from a *simulated* annotator into which we injected six kinds of errors plus one omission. The yellow badge says it is precomputed, not live." |
| 1:00–2:30 | **Review** | The top of the queue is the reversed and out-of-range items (rule, HIGH). Click **wipe the table with a sponge**: "The frames are static and the motion signal is flat. This is a hallucinated action." Mark **Incorrect** → delete. Click **move the red cube over the bin**: "Motion was already happening before its start, and the gap before it is flagged as a *candidate* missing action." Edit start 8.6 → 7.5 and save; show that the flags disappear on re-check. Click **grasp the red cube**: "The motion check is low-severity here because grasping only moves the gripper. That's a false alarm, and the flag says so." Mark **Correct** and resolve with the note "false alarm". |
| 2:30–3:00 | Interval findings | Show the 19–20 s candidate → "Add an annotation for this interval": "grasp the blue cube". |
| 3:00–3:40 | (optional, needs key) **Run model verifier** | "Rules can't catch a wrong label. Here a vision model reads start/middle/end frames: 'pick up the red cube' during a release is contradicted. It's tagged *model* because it's an opinion." Without a key, say this instead: "Wrong labels are invisible to rules; that's what the optional verifier is for." |
| 3:40–4:30 | **Dashboard** | Before/after counts, flag categories, the before→after diff table. Evaluation: "Because the sample has ground truth, we can measure QC: all six injected annotation errors are flagged, with two false alarms. That's precision 0.75 and recall 1.0 *on this synthetic sample only*, not a claim about real data." Show the flag codes per error, including the wrong label being caught only by coincidence. |
| 4:30–5:00 | **Export** | Download the QC report and review package: "Baseline, reviewed annotations, every flag and every reviewer decision, timestamped. That's an audit trail for a dataset." Close: "Next steps: real robot datasets with human references, and learned boundary refinement." |

Backup: if the public app is down, run locally with `streamlit run app.py`. The sample path needs no network.
