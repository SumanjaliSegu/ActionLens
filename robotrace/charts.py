"""Altair charts for the review timeline. Pure functions of session data."""
from __future__ import annotations

import altair as alt
import pandas as pd

from .schemas import ReviewSession

STATUS_COLORS = {"unreviewed": "#94a3b8", "correct": "#16a34a", "incorrect": "#dc2626", "needs_review": "#d97706"}
SEVERITY_COLORS = {"high": "#dc2626", "medium": "#d97706", "low": "#64748b"}


def timeline_chart(session: ReviewSession, selected_id: str | None = None, width: int | str = "container"):
    dur = session.video.duration_s
    rows = []
    for i, a in enumerate(sorted(session.annotations, key=lambda x: min(x.start, x.end))):
        lo, hi = sorted((a.start, a.end))
        st = session.status.get(a.id)
        rows.append({"lane": f"{i + 1:02d} {a.action[:34]}", "start": max(lo, 0), "end": min(max(hi, lo + 0.05), dur),
                     "action": a.action, "status": getattr(st, "value", st or "unreviewed"),
                     "interval": f"{a.start:.2f}–{a.end:.2f}s", "selected": a.id == selected_id,
                     "invalid": a.end <= a.start or a.end > dur + 0.05})
    df = pd.DataFrame(rows)
    x = alt.X("start:Q", scale=alt.Scale(domain=[0, dur]), title="time (s)")
    layers = []
    if not df.empty:
        layers.append(alt.Chart(df).mark_bar(cornerRadius=3, height=14).encode(
            x=x, x2="end:Q", y=alt.Y("lane:N", title=None, sort=None, axis=alt.Axis(labelLimit=220)),
            color=alt.Color("status:N", scale=alt.Scale(domain=list(STATUS_COLORS), range=list(STATUS_COLORS.values())),
                            legend=alt.Legend(orient="top", title=None)),
            stroke=alt.condition("datum.selected", alt.value("#0f172a"), alt.value(None)),
            strokeWidth=alt.condition("datum.selected", alt.value(2.5), alt.value(0)),
            strokeDash=alt.condition("datum.invalid", alt.value([3, 2]), alt.value([0])),
            tooltip=["action", "interval", "status"]))
    lane = alt.layer(*layers) if layers else alt.Chart(pd.DataFrame({"start": [0]})).mark_point(opacity=0).encode(x=x)

    fl = [{"start": f.start, "end": max(f.end, f.start + 0.08), "code": f.code, "severity": f.severity.value,
           "reason": f.reason[:160]}
          for f in session.flags if not f.resolved and f.start is not None and f.annotation_id is None]
    charts = [lane.properties(height=max(60, 22 * len(df)), width=width)]
    if fl:
        fdf = pd.DataFrame(fl)
        fdf["start"] = fdf["start"].clip(lower=0, upper=dur)
        fdf["end"] = fdf["end"].clip(lower=0, upper=dur)
        charts.append(alt.Chart(fdf).mark_rect(opacity=0.55, height=12).encode(
            x=alt.X("start:Q", scale=alt.Scale(domain=[0, dur]), title=None), x2="end:Q",
            y=alt.Y("code:N", title=None, axis=alt.Axis(labelLimit=220)),
            color=alt.Color("severity:N", scale=alt.Scale(domain=list(SEVERITY_COLORS),
                                                          range=list(SEVERITY_COLORS.values())), legend=None),
            tooltip=["code", "severity", "reason"]).properties(height=22 * fdf["code"].nunique() + 10, width=width))
    if session.signals:
        sdf = pd.DataFrame({"t": session.signals["times"], "motion": session.signals["motion"]})
        # Clip the axis so one spike (e.g. a lighting glitch) doesn't flatten the rest; tooltip keeps raw values.
        top = max(float(sdf["motion"].quantile(0.98)) * 1.2, 1e-3)
        charts.append(alt.Chart(sdf).mark_area(opacity=0.35, color="#0e7490", line={"color": "#0e7490"},
                                               clip=True).encode(
            x=alt.X("t:Q", scale=alt.Scale(domain=[0, dur]), title="time (s)"),
            y=alt.Y("motion:Q", title="motion", scale=alt.Scale(domain=[0, top], clamp=True)),
            tooltip=["t", "motion"]).properties(height=70, width=width))
    return alt.vconcat(*charts, spacing=4).resolve_scale(color="independent").configure_view(stroke=None)


def flag_category_chart(session: ReviewSession):
    rows = []
    for label, flags in (("before review", session.initial_flags), ("current", session.flags)):
        for f in flags:
            if label == "current" and f.resolved:
                continue
            rows.append({"when": label, "category": f.category})
    if not rows:
        return None
    df = pd.DataFrame(rows).value_counts().reset_index(name="count")
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("count:Q", title="open flags"), y=alt.Y("category:N", title=None),
        color=alt.Color("when:N", scale=alt.Scale(domain=["before review", "current"], range=["#94a3b8", "#0e7490"]),
                        legend=alt.Legend(orient="top", title=None)),
        yOffset="when:N", tooltip=["when", "category", "count"]).properties(height=200)
