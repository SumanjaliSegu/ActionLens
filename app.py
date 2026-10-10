"""RoboTrace Streamlit app. UI and workflow orchestration only; logic lives in the robotrace package.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from robotrace import __version__
from robotrace import export_service as ex
from robotrace import review_service as rs
from robotrace.annotation_engine import ADAPTERS, MUSE_DATA_NOTICE, AdapterUnavailable, get_adapter
from robotrace.charts import SEVERITY_COLORS, STATUS_COLORS, flag_category_chart, timeline_chart
from robotrace.config import SAMPLE_DIR, get_settings
from robotrace.evaluation import Segment, evaluate_flags, evaluate_segments
from robotrace.pipeline import SAMPLE_VIDEO, cache_dir_for, process_video, sample_metadata
from robotrace.quality_engine import PRIORITY_RULE, model_verify, rank_annotations
from robotrace.schemas import ReviewSession, ReviewStatus
from robotrace.storage import cleanup_old_files
from robotrace.video_processor import (ALLOWED_EXTENSIONS, VideoError, make_browser_playable, probe,
                                       validate_duration, validate_upload)

st.set_page_config(page_title="RoboTrace — robotics annotation QC", page_icon="🎯", layout="wide",
                   initial_sidebar_state="expanded")
SETTINGS = get_settings()
PAGES = ["Overview", "Analyze", "Review", "Dashboard", "Export"]

CSS = """
<style>
:root { --rt-ink:#0f172a; --rt-muted:#64748b; --rt-line:#e2e8f0; --rt-accent:#0e7490; --rt-soft:#f1f5f9; }
.block-container { padding-top: 3.2rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.rt-hero { padding: 1.6rem 1.8rem; border-radius: 14px; border:1px solid var(--rt-line);
  background: linear-gradient(135deg, #f8fafc 0%, #ecfeff 100%); }
.rt-hero h1 { font-size: 2.3rem; margin: 0 0 .2rem 0; color: var(--rt-ink); }
.rt-hero p.tag { font-size: 1.12rem; color: #334155; margin: 0 0 .6rem 0; }
.rt-kicker { font-size:.78rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color: var(--rt-accent); }
.rt-flow { display:flex; flex-wrap:wrap; gap:.5rem; align-items:stretch; margin:.6rem 0 0 0; }
.rt-step { flex:1 1 140px; border:1px solid var(--rt-line); border-radius:10px; padding:.7rem .8rem; background:#fff; }
.rt-step b { display:block; color:var(--rt-ink); font-size:.95rem; }
.rt-step span { color:var(--rt-muted); font-size:.82rem; }
.rt-step.core { border-color: var(--rt-accent); box-shadow: 0 0 0 1px var(--rt-accent) inset; }
.rt-arrow { align-self:center; color:#94a3b8; font-size:1.2rem; }
.rt-badge { display:inline-block; padding:.1rem .5rem; border-radius:999px; font-size:.74rem; font-weight:600;
  border:1px solid transparent; margin-right:.25rem; white-space:nowrap; }
.rt-live { background:#dcfce7; color:#166534; border-color:#86efac; }
.rt-sample { background:#fef3c7; color:#92400e; border-color:#fcd34d; }
.rt-high { background:#fee2e2; color:#991b1b; } .rt-medium { background:#ffedd5; color:#9a3412; }
.rt-low { background:#f1f5f9; color:#334155; }
.rt-origin { background:#eef2ff; color:#3730a3; }
.rt-mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:.85rem; }
.rt-note { border-left:3px solid #d97706; background:#fffbeb; padding:.6rem .8rem; border-radius:6px; color:#78350f; font-size:.9rem; }
.rt-metric { border:1px solid var(--rt-line); border-radius:10px; padding:.7rem .9rem; background:#fff; }
.rt-metric .v { font-size:1.6rem; font-weight:700; color:var(--rt-ink); }
.rt-metric .l { font-size:.8rem; color:var(--rt-muted); }
div[data-testid="stSidebarNav"] { display:none; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------- state
def S():
    return st.session_state


def init_state() -> None:
    S().setdefault("page", "Overview")
    S().setdefault("review", None)
    S().setdefault("player_path", None)
    S().setdefault("upload_path", None)
    S().setdefault("sel", None)
    S().setdefault("seek", 0)
    S().setdefault("reference", None)
    S().setdefault("flash", None)
    if not S().get("_cleaned"):
        cleanup_old_files(SETTINGS.work_dir / "uploads", max_age_hours=6)
        cleanup_old_files(SETTINGS.work_dir / "cache", max_age_hours=24)
        S()["_cleaned"] = True


def goto(page: str) -> None:
    S()["page"] = page


def flash(msg: str, kind: str = "success") -> None:
    S()["flash"] = (kind, msg)


def show_flash() -> None:
    f = S().get("flash")
    if f:
        kind, msg = f
        getattr(st, kind)(msg)
        S()["flash"] = None


def review() -> ReviewSession | None:
    return S().get("review")


_ESC = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;", "[": "&#91;", "]": "&#93;",
        "(": "&#40;", ")": "&#41;", "!": "&#33;", "*": "&#42;", "_": "&#95;", "`": "&#96;", "~": "&#126;",
        "\\": "&#92;", "#": "&#35;", "|": "&#124;", "$": "&#36;"}


def esc(text: str) -> str:
    """Escape untrusted text (model output, file names) for st.markdown / st.caption / alerts, with or without
    unsafe_allow_html: neutralises HTML and Markdown syntax (links, images, emphasis, math), so model output can
    neither inject markup nor make the viewer's browser load external URLs. Do not use inside `code spans`."""
    return "".join(_ESC.get(c, c) for c in str(text))


def mode_badge(sess: ReviewSession) -> str:
    p = sess.processing
    if p.is_precomputed_sample:
        return '<span class="rt-badge rt-sample">PRECOMPUTED SAMPLE</span>'
    if p.adapter == "motion":
        return '<span class="rt-badge rt-low">MOTION SEGMENTATION · NO AI</span>'
    if p.is_live_inference:
        who = f"{p.provider} · {p.model}" if p.provider and p.model else p.adapter
        return f'<span class="rt-badge rt-live">LIVE AI · {esc(who)}</span>'
    return f'<span class="rt-badge rt-low">{esc(p.adapter)}</span>'


def sev_badge(sev: str) -> str:
    return f'<span class="rt-badge rt-{sev}">{sev.upper()}</span>'


def origin_badge(origin: str) -> str:
    label = {"deterministic_rule": "rule", "visual_signal": "pixel signal", "model_judgment": "model"}[origin]
    return f'<span class="rt-badge rt-origin">{label}</span>'


def fmt_t(t: float) -> str:
    sign = "-" if t < 0 else ""
    t = abs(t)
    return f"{sign}{int(t // 60):d}:{t % 60:05.2f}"


# ----------------------------------------------------------------------------------------- loading
def load_sample() -> None:
    try:
        meta = sample_metadata()
        sess = process_video(SAMPLE_VIDEO, meta, "sample", settings=SETTINGS)
    except Exception as e:  # noqa: BLE001
        flash(f"Could not load the sample: {e}", "error")
        return
    S()["review"], S()["player_path"] = sess, str(SAMPLE_VIDEO)
    S()["sel"] = rank_annotations(sess.annotations, sess.flags)[0][0].id if sess.annotations else None
    S()["reference"] = load_sample_reference(meta.sha256)
    flash("Loaded the synthetic sample with precomputed (simulated) annotations.")
    goto("Review")


def load_sample_reference(sha: str) -> dict | None:
    gt_p, base_p = SAMPLE_DIR / "ground_truth.json", SAMPLE_DIR / "baseline_annotations.json"
    if not gt_p.exists():
        return None
    gt = json.loads(gt_p.read_text())
    if gt.get("video_sha256") != sha:
        return None
    base = json.loads(base_p.read_text())
    return {"name": "synthetic sample ground truth (from renderer)", "segments": gt["segments"],
            "erroneous_ids": [a["id"] for a in base["annotations"] if a.get("injected_error")],
            "error_types": {a["id"]: a["injected_error"] for a in base["annotations"] if a.get("injected_error")},
            "missing": base.get("missing_segments", [])}


# ------------------------------------------------------------------------------------------ sidebar
def sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🎯 RoboTrace")
        st.caption("Evidence-grounded robotics video annotation & QC")
        st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
        st.divider()
        sess = review()
        if sess:
            st.markdown(f"**Current video**  \n<span class='rt-mono'>{esc(sess.video.filename)}</span>",
                        unsafe_allow_html=True)
            st.markdown(mode_badge(sess), unsafe_allow_html=True)
            open_flags = sum(not f.resolved for f in sess.flags)
            reviewed = sum(1 for a in sess.annotations if sess.status.get(a.id) != ReviewStatus.unreviewed)
            st.caption(f"{len(sess.annotations)} annotations · {open_flags} open flags · "
                       f"{reviewed}/{len(sess.annotations)} reviewed")
        else:
            st.caption("No video loaded yet.")
        st.divider()
        with st.expander("Engine status", expanded=False):
            for name in ADAPTERS:
                ok, why = get_adapter(name).availability()
                st.markdown(f"{'🟢' if ok else '⚪'} **{name}** — {why}")
        st.caption(f"v{__version__} · state is per browser session and is not persisted on the server.")


# ----------------------------------------------------------------------------------------- overview
def page_overview() -> None:
    st.markdown("""
<div class="rt-hero">
  <div class="rt-kicker">Physical AI · robotics data quality</div>
  <h1>RoboTrace</h1>
  <p class="tag">Find the annotation errors in robot demonstration videos before they reach your training set.</p>
  <p style="color:#475569;margin:0">RoboTrace generates or ingests timestamped action annotations, checks every one
  against the video's own evidence, and hands reviewers a ranked queue of suspicious segments, each with the
  reason it was flagged.</p>
</div>""", unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.button("▶ Try the sample", type="primary", use_container_width=True, on_click=load_sample,
              help="Synthetic pick-and-place clip with precomputed annotations. No GPU or API key needed.")
    c2.button("⬆ Upload a video", use_container_width=True, on_click=goto, args=("Analyze",))
    show_flash()

    st.markdown("#### How the pipeline works")
    st.markdown("""
<div class="rt-flow">
  <div class="rt-step"><b>1 · Ingest</b><span>Validate format, size, duration; decode with OpenCV; hash for caching.</span></div>
  <div class="rt-arrow">→</div>
  <div class="rt-step"><b>2 · Annotate</b><span>Swappable adapter: Gemini or Claude vision, remote GPU video-LLM, or motion segmentation.</span></div>
  <div class="rt-arrow">→</div>
  <div class="rt-step core"><b>3 · Quality control</b><span>Timestamp rules · pixel-signal evidence · optional model verifier.</span></div>
  <div class="rt-arrow">→</div>
  <div class="rt-step"><b>4 · Human review</b><span>Ranked queue, evidence frames, edit / accept / reject, resolve flags.</span></div>
  <div class="rt-arrow">→</div>
  <div class="rt-step"><b>5 · Export</b><span>JSON, CSV, QC report, full review package with decision log.</span></div>
</div>""", unsafe_allow_html=True)

    st.markdown("#### What the quality-control layer checks")
    a, b, c = st.columns(3)
    with a, st.container(border=True):
        st.markdown("**Deterministic rules**")
        st.caption("Negative, reversed, zero-length, out-of-range, and overlapping intervals; unusually long or "
                   "short segments; uncovered gaps. Pure arithmetic: always reproducible.")
    with b, st.container(border=True):
        st.markdown("**Visual-signal evidence**")
        st.caption("Frame-difference motion energy and histogram change: actions with no visible motion, "
                   "idle stretches inside segments, boundaries that cut through ongoing motion, unexplained "
                   "abrupt changes, and uncovered motion (possible missing actions).")
    with c, st.container(border=True):
        st.markdown("**Model judgment (optional)**")
        st.caption("A vision-language model looks at each segment's start/middle/end frames and says whether "
                   "they support the description. Shown separately and labelled as an opinion.")

    st.markdown("#### Honest notes")
    st.markdown("""
<div class="rt-note">
• Flags are <b>prompts for a human</b>, not verdicts. "Possible missing action" means unexplained motion, not a confirmed omission.<br>
• Review priority is a transparent weighted count of flags, <b>not a calibrated probability</b> of error.<br>
• The bundled sample is a <b>synthetic rendering</b> with a <b>simulated</b> annotator whose errors were injected on purpose, so QC results on it can be checked against known truth. Numbers on the sample say nothing about performance on real robot footage.<br>
• Live Claude annotation sees sampled still frames, not continuous video, so boundaries are approximate.
</div>""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------ analyze
def save_upload(uploaded) -> Path:
    validate_upload(uploaded.name, uploaded.size, SETTINGS)
    up_dir = SETTINGS.work_dir / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    old = S().get("upload_path")
    if old:
        Path(old).unlink(missing_ok=True)
    # Never use the user-supplied name as a path component.
    dest = up_dir / f"{uuid.uuid4().hex}{Path(uploaded.name).suffix.lower()}"
    with open(dest, "wb") as f:
        f.write(uploaded.getbuffer())
    S()["upload_path"] = str(dest)
    return dest


def page_analyze() -> None:
    st.markdown("## Analyze a video")
    show_flash()
    source = st.radio("Source", ["Upload my video", "Bundled synthetic sample"], horizontal=True)
    if source == "Bundled synthetic sample":
        st.info("The sample opens with precomputed, clearly-labelled simulated annotations. You can also run a "
                "live engine on it below to compare.")
    uploaded = None
    if source == "Upload my video":
        uploaded = st.file_uploader(
            f"Video file ({', '.join(sorted(e[1:] for e in ALLOWED_EXTENSIONS))}; ≤ {SETTINGS.max_upload_mb:.0f} MB, "
            f"≤ {SETTINGS.max_duration_s:.0f} s)", type=[e[1:] for e in ALLOWED_EXTENSIONS])

    st.markdown("#### Annotation engine")
    options = ["muse", "gemini", "claude", "remote", "motion"] +(["sample"] if source != "Upload my video" else [])
    avail = {n: get_adapter(n).availability() for n in options}
    default = next((i for i, n in enumerate(options) if avail[n][0]), len(options) - 1)
    if source != "Upload my video":
        default = options.index("sample")
    engine = st.radio("Engine", options, index=default, horizontal=True,
                      format_func=lambda n: f"{ADAPTERS[n].label}{'' if avail[n][0] else ' — unavailable'}")
    ok, why = avail[engine]
    (st.caption if ok else st.warning)(why)
    if engine in ("muse", "gemini"):
        from robotrace.sampling import describe_sampling, sampling_config
        known = sample_metadata().duration_s if source != "Upload my video" else None
        st.caption("Planned: " + describe_sampling(sampling_config(SETTINGS), known)
                   + ("" if known else " (exact counts depend on the video's length)") + ".")
    if engine == "muse" and "contributor" in SETTINGS.muse_model:
        st.info(MUSE_DATA_NOTICE + " Only analyze videos you are allowed to share on those terms.")
    model_check_ok = SETTINGS.has_anthropic_key
    model_check = st.checkbox("Also run the model-based evidence verifier (Claude vision; one extra API call)",
                              value=False, disabled=not model_check_ok,
                              help=None if model_check_ok else "Requires ANTHROPIC_API_KEY.")

    run = st.button("Run analysis", type="primary", disabled=not ok or (source == "Upload my video" and uploaded is None))
    if not run:
        return
    progress = st.progress(0.0, text="Preparing…")
    try:
        if source == "Upload my video":
            path = save_upload(uploaded)
            meta = probe(path, display_name=Path(uploaded.name).name[:120])
            validate_duration(meta, SETTINGS)
        else:
            path, meta = SAMPLE_VIDEO, sample_metadata()
        sess = process_video(path, meta, engine, model_check=model_check, settings=SETTINGS,
                             progress=lambda p, m: progress.progress(min(p, 1.0) * 0.9, text=m))
        progress.progress(0.95, text="Preparing browser-playable video…")
        player = (str(path) if meta.is_synthetic_sample
                  else str(make_browser_playable(path, cache_dir_for(meta, SETTINGS) / "playable.mp4")))
    except VideoError as e:
        progress.empty()
        if S().get("upload_path"):
            Path(S()["upload_path"]).unlink(missing_ok=True)
            S()["upload_path"] = None
        st.error(f"Video rejected: {e}")
        return
    except AdapterUnavailable as e:
        progress.empty()
        st.error(f"Annotation engine '{engine}' failed: {e}  \nNo annotations were generated and nothing was "
                 "substituted. Fix the configuration and retry, or pick another engine (motion segmentation "
                 "needs no API).")
        return
    except Exception as e:  # noqa: BLE001 - surface unexpected errors without a traceback dump
        progress.empty()
        st.error(f"Processing failed ({type(e).__name__}): {e}")
        return
    progress.progress(1.0, text="Done")
    S()["review"], S()["player_path"] = sess, player
    S()["sel"] = rank_annotations(sess.annotations, sess.flags)[0][0].id if sess.annotations else None
    S()["reference"] = load_sample_reference(meta.sha256) if meta.is_synthetic_sample else None
    n_open = sum(not f.resolved for f in sess.flags)
    flash(f"Processed {meta.filename}: {len(sess.annotations)} annotations, {n_open} flags.")
    engine_report(sess)
    for w in sess.processing.warnings:
        st.warning(esc(w))
    if not sess.annotations:
        st.warning("The engine returned no annotations. You can still add annotations manually in Review.")
    st.button("Open review →", type="primary", on_click=goto, args=("Review",))


def engine_report(sess: ReviewSession) -> None:
    """Which engine actually ran, and (for vision models) what it reported seeing."""
    p = sess.processing
    st.markdown(f"#### Engine that ran &nbsp; {mode_badge(sess)}", unsafe_allow_html=True)
    if p.provider:
        st.markdown(f"**Provider:** {esc(p.provider)} &nbsp;·&nbsp; **Model:** <span class='rt-mono'>"
                    f"{esc(p.model)}</span>",
                    unsafe_allow_html=True)
    st.caption(f"{ADAPTERS[p.adapter].label if p.adapter in ADAPTERS else esc(p.adapter)} · {esc(p.adapter_detail)} · "
               f"annotation took {p.annotation_seconds:.1f}s")
    if p.model_summary:
        st.markdown(f"**Model's scene description:** {esc(p.model_summary)}", unsafe_allow_html=True)
    if any(a.visual_evidence or a.objects for a in sess.baseline):
        st.markdown("**Model observations** (boundaries are approximate; confidence is self-reported)")
        st.dataframe(pd.DataFrame([{
            "start (s)": a.start, "end (s)": a.end, "± (s)": a.boundary_precision_s, "action": a.action,
            "objects": ", ".join(a.objects), "visual evidence": a.visual_evidence,
            "cited frames (s)": ", ".join(f"{t:.2f}" for t in a.cited_frame_times),
            "self-rated confidence (uncalibrated)": a.confidence} for a in sess.baseline]),
            hide_index=True, use_container_width=True)
    st.caption(f"Deterministic QC ran on these annotations: {len(sess.flags)} flag(s). "
               "Resolve them on the Review page.")


# ------------------------------------------------------------------------------------------- review
def select(ann_id: str, start: float) -> None:
    S()["sel"] = ann_id
    S()["seek"] = int(max(0, start))


def seek(t: float) -> None:
    S()["seek"] = int(max(0, t))


def page_review() -> None:
    sess = review()
    if not sess:
        empty_state()
        return
    st.markdown(f"## Review &nbsp; {mode_badge(sess)}", unsafe_allow_html=True)
    v = sess.video
    st.caption(f"{v.filename} · {v.duration_s:.2f}s · {v.fps:g} fps · {v.width}×{v.height} · "
               f"engine: {sess.processing.adapter_detail or sess.processing.adapter}")
    if sess.processing.is_precomputed_sample:
        st.markdown('<div class="rt-note">These annotations are <b>precomputed and simulated</b> (deliberate errors '
                    'injected for demonstration). They were not produced by live inference.</div>',
                    unsafe_allow_html=True)
    if sess.processing.model_summary:
        st.caption(f"Model's scene description: {esc(sess.processing.model_summary)}")
    show_flash()

    left, right = st.columns([3, 2], gap="large")
    with left:
        if S().get("player_path") and Path(S()["player_path"]).exists():
            st.video(S()["player_path"], start_time=int(S().get("seek", 0)))
            st.caption(f"Player positioned at {fmt_t(S().get('seek', 0))} (whole-second seek). "
                       "Select an item to jump to it.")
        else:
            st.warning("Video file is no longer available on the server; evidence frames are still shown.")
        st.altair_chart(timeline_chart(sess, S().get("sel")), use_container_width=True)
    with right:
        review_queue(sess)

    st.divider()
    sel = S().get("sel")
    if sel and any(a.id == sel for a in sess.annotations):
        annotation_detail(sess, sess.get(sel))
    unassigned_findings(sess)
    tools_row(sess)


def empty_state(title: str = "Review") -> None:
    st.markdown(f"## {title}")
    st.info("Nothing to review yet. Load the sample or analyze a video first.")
    c1, c2, _ = st.columns([1, 1, 3])
    c1.button("▶ Load sample", type="primary", on_click=load_sample, use_container_width=True)
    c2.button("⬆ Upload", on_click=goto, args=("Analyze",), use_container_width=True)


def review_queue(sess: ReviewSession) -> None:
    st.markdown("#### Review queue")
    st.caption(PRIORITY_RULE)
    f1, f2 = st.columns(2)
    only_flagged = f1.toggle("Only flagged", value=False)
    status_filter = f2.selectbox("Status", ["all"] + [s.value for s in ReviewStatus], label_visibility="collapsed")
    rows = rank_annotations(sess.annotations, sess.flags)
    shown = 0
    with st.container(height=520):
        for a, score, flags in rows:
            st_val = sess.status.get(a.id, ReviewStatus.unreviewed).value
            if only_flagged and score == 0:
                continue
            if status_filter != "all" and st_val != status_filter:
                continue
            shown += 1
            open_f = [f for f in flags if not f.resolved]
            worst = min((f.severity.value for f in open_f), key=["high", "medium", "low"].index, default=None)
            dot = {"high": "🔴", "medium": "🟠", "low": "⚪"}.get(worst, "🟢")
            stat = {"correct": "✓", "incorrect": "✗", "needs_review": "?", "unreviewed": "·"}[st_val]
            label = f"{dot} {stat}  {a.start:6.2f}–{a.end:6.2f}s  {a.action[:44]}"
            if score:
                label += f"  · P{score}"
            st.button(label, key=f"q_{a.id}", use_container_width=True,
                      type="primary" if a.id == S().get("sel") else "secondary",
                      on_click=select, args=(a.id, min(a.start, a.end)),
                      help=" | ".join(f"{f.code}" for f in open_f) or "No open flags")
        if shown == 0:
            st.caption("No annotations match the current filter.")


def annotation_detail(sess: ReviewSession, a) -> None:
    status = sess.status.get(a.id, ReviewStatus.unreviewed)
    st.markdown(f"### {esc(a.action)}", unsafe_allow_html=True)
    meta_bits = [f"`{a.id}`", f"{a.start:.2f}s → {a.end:.2f}s ({a.end - a.start:.2f}s)", f"source: {a.source}",
                 f"status: **{status.value}**"]
    if a.confidence is not None:
        meta_bits.append(f"model self-reported confidence {a.confidence:.2f} (uncalibrated)")
    if a.boundary_precision_s is not None:
        meta_bits.append(f"boundaries approximate ±{a.boundary_precision_s:.2f}s")
    st.markdown(" · ".join(meta_bits))
    if a.objects or a.visual_evidence:
        cited = ", ".join(f"{t:.2f}s" for t in a.cited_frame_times) or "none"
        st.markdown(f"**Objects:** {esc(', '.join(a.objects) or '—')} &nbsp;·&nbsp; **Model's visual evidence** "
                    f"(original model output): {esc(a.visual_evidence or '—')} · cited frames: {cited}",
                    unsafe_allow_html=True)
        st.caption("Object names, colours and boundaries come from a vision model and may be wrong. Check them "
                   "against the evidence frames and correct them in the edit form.")

    c_ev, c_act = st.columns([3, 2], gap="large")
    with c_ev:
        st.markdown("**Evidence frames**")
        if a.evidence:
            cols = st.columns(len(a.evidence))
            for col, f in zip(cols, a.evidence):
                if f.path and Path(f.path).exists():
                    col.image(f.path, caption=f"t = {f.t:.2f}s (frame {f.frame_index})", use_column_width=True)
                col.button("Jump", key=f"jump_{a.id}_{f.frame_index}", on_click=seek, args=(f.t,))
        else:
            st.caption("No evidence frames (interval lies outside the video).")
        verdict = sess.model_verdicts.get(a.id)
        if verdict:
            st.markdown(f"**Model verifier ({esc(verdict.get('model', ''))}):** `{verdict['verdict']}` — "
                        f"{esc(verdict['observed'])}", unsafe_allow_html=True)

        st.markdown("**Flags**")
        mine = [f for f in sess.flags if f.annotation_id == a.id]
        if not mine:
            st.caption("No flags for this annotation.")
        for f in sorted(mine, key=lambda f: (f.resolved, ["high", "medium", "low"].index(f.severity.value))):
            flag_card(sess, f)

    with c_act:
        st.markdown("**Decision**")
        b1, b2, b3 = st.columns(3)
        for col, (val, label) in zip((b1, b2, b3), ((ReviewStatus.correct, "✓ Correct"),
                                                    (ReviewStatus.incorrect, "✗ Incorrect"),
                                                    (ReviewStatus.needs_review, "? Needs review"))):
            if col.button(label, key=f"st_{val.value}_{a.id}", use_container_width=True,
                          type="primary" if status == val else "secondary"):
                rs.set_status(sess, a.id, val)
                st.rerun()
        with st.form(f"edit_{a.id}", border=True):
            st.markdown("**Edit annotation**")
            dur = sess.video.duration_s
            e1, e2 = st.columns(2)
            new_start = e1.number_input("Start (s)", value=float(a.start), step=0.1, format="%.2f")
            new_end = e2.number_input("End (s)", value=float(a.end), step=0.1, format="%.2f")
            new_action = st.text_input("Action description", value=a.action, max_chars=300)
            new_objects = st.text_input("Objects (comma-separated)", value=", ".join(a.objects), max_chars=400,
                                        help="Model-reported object names and colours can be wrong; "
                                             "correct them here.")
            note = st.text_input("Reviewer note (optional)", value=sess.notes.get(a.id, ""), max_chars=300)
            mark = st.checkbox("Mark as correct after saving", value=True)
            if st.form_submit_button("Save changes", type="primary", use_container_width=True):
                try:
                    rs.edit_annotation(sess, a.id, start=new_start, end=new_end, action=new_action,
                                       objects=[o for o in new_objects.split(",")])
                    if note != sess.notes.get(a.id, ""):
                        sess.notes[a.id] = note
                    if mark:
                        rs.set_status(sess, a.id, ReviewStatus.correct, note)
                    flash("Saved. QC re-ran on the edited annotations.")
                    st.rerun()
                except rs.ReviewError as e:
                    st.error(str(e))
            st.caption(f"Valid range: 0 – {dur:.2f}s. Saving re-runs QC; flags that no longer apply disappear.")
        with st.expander("Delete annotation"):
            confirm = st.checkbox("I want to delete this annotation", key=f"delc_{a.id}")
            if st.button("Delete", key=f"del_{a.id}", disabled=not confirm):
                rs.delete_annotation(sess, a.id)
                remaining = rank_annotations(sess.annotations, sess.flags)
                S()["sel"] = remaining[0][0].id if remaining else None
                flash("Annotation deleted (kept in the baseline and the export's deletion list).")
                st.rerun()


def flag_card(sess: ReviewSession, f) -> None:
    with st.container(border=True):
        state = "✅ resolved" if f.resolved else ""
        st.markdown(f"{sev_badge(f.severity.value)}{origin_badge(f.origin.value)}"
                    f"<span class='rt-mono'>{f.code}</span> {state}<br>"
                    f"<span style='font-size:.9rem'>{esc(f.reason)}</span>", unsafe_allow_html=True)
        if f.details:
            with st.expander("Raw values"):
                st.json({k: v for k, v in f.details.items()}, expanded=True)
        if f.resolved:
            st.caption(f"Resolution: {f.resolution_note or '—'}")
            if st.button("Reopen", key=f"reopen_{f.id}"):
                rs.resolve_flag(sess, f.id, resolved=False)
                st.rerun()
        else:
            n1, n2 = st.columns([3, 1])
            note = n1.text_input("Resolution note", key=f"note_{f.id}", placeholder="e.g. verified, false alarm",
                                 label_visibility="collapsed")
            if n2.button("Resolve", key=f"res_{f.id}", use_container_width=True):
                rs.resolve_flag(sess, f.id, note or "resolved")
                st.rerun()


def unassigned_findings(sess: ReviewSession) -> None:
    items = [f for f in sess.flags if f.annotation_id is None]
    st.markdown("#### Interval findings")
    st.caption("Findings not tied to a single annotation: uncovered gaps, possible missing actions (candidates), "
               "and unexplained abrupt changes.")
    if not items:
        st.caption("None.")
        return
    for f in sorted(items, key=lambda f: (f.resolved, f.start or 0)):
        with st.container(border=True):
            c1, c2 = st.columns([4, 2])
            with c1:
                st.markdown(f"{sev_badge(f.severity.value)}{origin_badge(f.origin.value)}"
                            f"<span class='rt-mono'>{f.code}</span> {'✅ resolved' if f.resolved else ''}<br>"
                            f"<span style='font-size:.9rem'>{esc(f.reason)}</span>", unsafe_allow_html=True)
            with c2:
                st.button(f"Jump to {f.start:.2f}s", key=f"jf_{f.id}", on_click=seek, args=(f.start,))
                if not f.resolved:
                    if st.button("Dismiss", key=f"dis_{f.id}", help="Resolve as not an issue"):
                        rs.resolve_flag(sess, f.id, "dismissed by reviewer")
                        st.rerun()
            if not f.resolved and f.category in ("missing_candidate", "timestamp") and f.end and f.end > f.start:
                with st.expander("Add an annotation for this interval"):
                    with st.form(f"add_{f.id}"):
                        a1, a2 = st.columns(2)
                        s0 = a1.number_input("Start (s)", value=float(f.start), step=0.1, format="%.2f")
                        e0 = a2.number_input("End (s)", value=float(f.end), step=0.1, format="%.2f")
                        txt = st.text_input("Action description", max_chars=300)
                        if st.form_submit_button("Add annotation", type="primary"):
                            try:
                                new = rs.add_annotation(sess, s0, e0, txt, from_flag=f.id)
                                S()["sel"] = new.id
                                flash("Annotation added and marked correct; QC re-ran.")
                                st.rerun()
                            except rs.ReviewError as e:
                                st.error(str(e))


def tools_row(sess: ReviewSession) -> None:
    st.markdown("#### Tools")
    c1, c2, c3 = st.columns(3)
    if c1.button("Re-run QC", use_container_width=True):
        rs.recheck(sess)
        flash("QC re-ran on the current annotations.")
        st.rerun()
    can_verify = SETTINGS.has_anthropic_key and sess.video.filename and bool(sess.annotations)
    if c2.button("Run model verifier", use_container_width=True, disabled=not can_verify,
                 help=None if SETTINGS.has_anthropic_key else "Requires ANTHROPIC_API_KEY"):
        with st.spinner("Asking the vision model to check each annotation's frames…"):
            try:
                path = Path(S()["player_path"]) if S().get("player_path") else None
                missing_ev = [a for a in sess.annotations if not a.evidence and a.end > a.start]
                if missing_ev and path and path.exists():
                    from robotrace.annotation_engine import attach_evidence
                    attach_evidence(missing_ev, path, sess.video, cache_dir_for(sess.video, SETTINGS))
                sess.model_verdicts.update(model_verify(sess.annotations, SETTINGS))
                rs.recheck(sess)
                flash("Model verification complete. Model flags are tagged 'model'.")
            except AdapterUnavailable as e:
                flash(f"Verifier unavailable: {e}", "error")
        st.rerun()
    if c3.button("Reset review to engine output", use_container_width=True):
        S()["confirm_reset"] = True
    if S().get("confirm_reset"):
        st.warning("This discards all edits, decisions and resolutions.")
        y, n, _ = st.columns([1, 1, 4])
        if y.button("Yes, reset", type="primary"):
            sess.annotations = [a.model_copy(deep=True) for a in sess.baseline]
            sess.status = {a.id: ReviewStatus.unreviewed for a in sess.annotations}
            sess.notes, sess.events, sess.deleted_ids = {}, [], []
            sess.flags = [f.model_copy(deep=True) for f in sess.initial_flags]
            S()["confirm_reset"] = False
            st.rerun()
        if n.button("Cancel"):
            S()["confirm_reset"] = False
            st.rerun()


# ---------------------------------------------------------------------------------------- dashboard
def metric(col, value, label) -> None:
    col.markdown(f"<div class='rt-metric'><div class='v'>{value}</div><div class='l'>{label}</div></div>",
                 unsafe_allow_html=True)


def page_dashboard() -> None:
    sess = review()
    if not sess:
        empty_state("Dashboard")
        return
    s = rs.summary(sess)
    st.markdown(f"## Dashboard &nbsp; {mode_badge(sess)}", unsafe_allow_html=True)
    cols = st.columns(6)
    metric(cols[0], s["annotations_current"], f"annotations (baseline {s['annotations_baseline']})")
    metric(cols[1], s["flags_current"]["flagged_annotations"],
           f"flagged annotations (initially {s['flags_initial']['flagged_annotations']})")
    metric(cols[2], s["flags_current"]["open"], f"open flags (initially {s['flags_initial']['open']})")
    metric(cols[3], f"{s['reviewed']}/{s['annotations_current']}", "reviewed")
    metric(cols[4], s["review_action_total"], "reviewer actions logged")
    metric(cols[5], f"{s['annotation_seconds'] + s['qc_seconds']:.1f}s",
           f"processing (annot. {s['annotation_seconds']:.1f}s, QC {s['qc_seconds']:.1f}s)")
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Flags by category")
        ch = flag_category_chart(sess)
        if ch is not None:
            st.altair_chart(ch, use_container_width=True)
        else:
            st.caption("No flags.")
    with c2:
        st.markdown("#### Review status")
        sdf = pd.DataFrame([{"status": k, "count": v} for k, v in s["status"].items()])
        import altair as alt
        st.altair_chart(alt.Chart(sdf).mark_bar().encode(
            x=alt.X("count:Q"), y=alt.Y("status:N", sort=list(STATUS_COLORS), title=None),
            color=alt.Color("status:N", scale=alt.Scale(domain=list(STATUS_COLORS),
                                                        range=list(STATUS_COLORS.values())), legend=None)
        ).properties(height=200), use_container_width=True)

    st.markdown("#### Annotations")
    rows = []
    for a, score, flags in rank_annotations(sess.annotations, sess.flags):
        open_f = [f for f in flags if not f.resolved]
        rows.append({"priority": score, "start_s": a.start, "end_s": a.end, "action": a.action,
                     "status": sess.status.get(a.id, ReviewStatus.unreviewed).value,
                     "open_flags": len(open_f), "codes": ", ".join(sorted({f.code for f in open_f})),
                     "source": a.source, "id": a.id})
    df = pd.DataFrame(rows)
    if not df.empty:
        f1, f2, f3 = st.columns(3)
        stat = f1.multiselect("Status", sorted(df["status"].unique()), default=sorted(df["status"].unique()))
        codes = sorted({c for cs in df["codes"] for c in cs.split(", ") if c})
        code = f2.selectbox("Has flag", ["any"] + codes)
        text = f3.text_input("Search action text")
        m = df["status"].isin(stat)
        if code != "any":
            m &= df["codes"].str.contains(code, regex=False)
        if text:
            m &= df["action"].str.contains(text, case=False, regex=False)
        st.dataframe(df[m], use_container_width=True, hide_index=True, column_config={
            "priority": st.column_config.NumberColumn(help=PRIORITY_RULE),
            "start_s": st.column_config.NumberColumn(format="%.2f"),
            "end_s": st.column_config.NumberColumn(format="%.2f")})

    st.markdown("#### Before vs after review")
    before = {a.id: a for a in sess.baseline}
    after = {a.id: a for a in sess.annotations}
    diff = []
    for aid in list(before) + [i for i in after if i not in before]:
        b, a = before.get(aid), after.get(aid)
        change = ("added by reviewer" if b is None else "deleted" if a is None else
                  "edited" if (b.start, b.end, b.action) != (a.start, a.end, a.action) else "unchanged")
        diff.append({"id": aid, "change": change,
                     "before": f"{b.start:.2f}–{b.end:.2f}  {b.action}" if b else "—",
                     "after": f"{a.start:.2f}–{a.end:.2f}  {a.action}" if a else "—",
                     "decision": sess.status.get(aid, ReviewStatus.unreviewed).value})
    st.dataframe(pd.DataFrame(diff), use_container_width=True, hide_index=True)

    evaluation_section(sess)


def evaluation_section(sess: ReviewSession) -> None:
    st.markdown("#### Evaluation against reference annotations")
    ref = S().get("reference")
    up = st.file_uploader("Load reference annotations (JSON: {\"segments\": [{start, end, action}], optional "
                          "\"erroneous_ids\", \"missing\"})", type=["json"], key="ref_upload")
    if up is not None:
        try:
            data = json.loads(up.getvalue().decode("utf-8"))  # json only: no unsafe deserialisation
            segs = [Segment(float(s["start"]), float(s["end"]), str(s["action"])) for s in data["segments"]]
            ref = {"name": up.name, "segments": [s.__dict__ for s in segs],
                   "erroneous_ids": list(map(str, data.get("erroneous_ids", []))),
                   "missing": data.get("missing", []), "error_types": data.get("error_types", {})}
            S()["reference"] = ref
        except (ValueError, KeyError, TypeError) as e:
            st.error(f"Could not read reference file: {e}")
    if not ref:
        st.caption("No reference loaded, so no accuracy, recall or improvement numbers are shown.")
        return
    st.caption(f"Reference: **{ref['name']}**. Matching: greedy one-to-one at IoU ≥ 0.5; label match is "
               "normalised exact text.")
    segs = [Segment(**s) for s in ref["segments"]]
    rows = []
    for label, anns in (("Baseline (engine output)", sess.baseline), ("After review", sess.annotations)):
        e = evaluate_segments(anns, segs)
        rows.append({"annotations": label, "precision": e.precision, "recall": e.recall, "mean IoU": e.mean_iou,
                     "start err (s)": e.mean_start_error_s, "end err (s)": e.mean_end_error_s,
                     "label acc.": e.label_accuracy, "matched": f"{len(e.matches)}/{e.n_ref}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if ref.get("erroneous_ids") is not None and ref.get("erroneous_ids"):
        fe = evaluate_flags(sess.initial_flags, sess.baseline, set(ref["erroneous_ids"]),
                            [Segment(**m) for m in ref.get("missing", [])])
        st.markdown(f"**Initial QC flags vs known errors:** flag precision **{fe['flag_precision']}**, "
                    f"recall **{fe['flag_recall']}** ({fe['true_positive']} TP, {fe['false_positive']} FP, "
                    f"{fe['false_negative']} FN); missing segments with a gap/candidate flag: "
                    f"{fe['missing_with_candidate_or_gap_flag']}/{fe['missing_segments']}.")
        types = ref.get("error_types", {})
        st.dataframe(pd.DataFrame([{"annotation": k, "injected error": types.get(k, "?"), "flagged": bool(v),
                                    "flag codes": ", ".join(v) or "—"}
                                   for k, v in fe["per_error_caught"].items()]),
                     use_container_width=True, hide_index=True)
        st.caption("Read the codes, not just the tick: an error can be flagged by a rule that is not about its "
                   "actual defect (e.g. a wrong label caught only by a low-motion flag). Wrong labels can only be "
                   "detected deliberately by the model verifier.")
    if sess.video.is_synthetic_sample:
        st.caption("⚠ Computed on the synthetic sample with injected errors. This demonstrates the evaluation "
                   "utility; it is not evidence of performance on real robot data.")


# ------------------------------------------------------------------------------------------- export
def page_export() -> None:
    sess = review()
    if not sess:
        empty_state("Export")
        return
    st.markdown(f"## Export &nbsp; {mode_badge(sess)}", unsafe_allow_html=True)
    st.caption("All timestamps are in seconds from the start of the video. Schema: `robotrace.v1` "
               "(see docs/SCHEMA.md). Evidence is exported as timestamps and frame indices.")
    stem = Path(sess.video.filename).stem[:60] or "video"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    items = [
        ("Annotations (JSON)", "Current annotations with review status and notes.", ex.annotations_json(sess),
         f"{stem}_annotations_{stamp}.json", "application/json"),
        ("Annotations (CSV)", "One row per annotation with open flag codes.", ex.annotations_csv(sess),
         f"{stem}_annotations_{stamp}.csv", "text/csv"),
        ("QC report (Markdown)", "Summary, ranked review queue, every flag with its reason.",
         ex.qc_report_markdown(sess), f"{stem}_qc_report_{stamp}.md", "text/markdown"),
        ("Review package (JSON)", "Baseline + reviewed annotations, initial/current flags, model verdicts, "
         "full reviewer event log.", ex.review_package_json(sess), f"{stem}_review_package_{stamp}.json",
         "application/json"),
    ]
    for title, desc, data, fname, mime in items:
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{title}**  \n{desc}")
            c2.download_button("Download", data=data, file_name=fname, mime=mime, use_container_width=True,
                               key=f"dl_{fname}")
    with st.expander("Preview QC report"):
        st.markdown(ex.qc_report_markdown(sess))


# --------------------------------------------------------------------------------------------- main
def main() -> None:
    init_state()
    sidebar()
    {"Overview": page_overview, "Analyze": page_analyze, "Review": page_review,
     "Dashboard": page_dashboard, "Export": page_export}[S()["page"]]()


main()
