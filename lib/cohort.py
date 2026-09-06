"""Cross-participant statistics for the admin.

Turns `db.list_participant_summaries()` into a dataframe and renders:
  - overall distributions of the outcome + behaviour metrics
  - the same metrics split by delegation condition (strip + box)
  - manipulation checks (did the condition actually change behaviour?)
  - a per-condition n / mean / sd table with a hand-rolled one-way ANOVA

No SciPy dependency — the F-distribution p-value uses the regularized incomplete
beta function, in the same spirit as build_report.py hand-rolling PCA/k-means.
"""

import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CONDITIONS = ["slow_planning", "slow_iterating", "control"]
CONDITION_LABEL = {
    "slow_planning": "Slow planning",
    "slow_iterating": "Slow iterating",
    "control": "Control",
}

# categorical palette (matches session_timeline / build_report series colors)
CONDITION_COLOR = {
    "slow_planning": "#2a78d6",
    "slow_iterating": "#eb6834",
    "control": "#1baf7a",
}

# (column, human label) — the metrics worth comparing across conditions.
OUTCOME_METRICS = [
    ("notebook_pct", "Notebook score (%)"),
    ("quiz_pct", "Quiz score (%)"),
]
# Only the columns present in the data are charted, so listing both the
# transcript metrics and the history-file (prompt-only) equivalents is fine.
BEHAVIOUR_METRICS = [
    ("session_duration_s", "Session duration (s)"),
    ("n_tool_calls", "Tool calls"),
    ("n_edits", "File edits"),
    ("n_prompts", "Prompts"),
    ("n_sessions", "Sessions (context resets)"),
    ("time_to_first_tool_call_s", "Time to first tool call (s)"),
    ("time_to_first_prompt_s", "Time to first prompt (s)"),
    ("median_inter_tool_gap_s", "Median gap between tool calls (s)"),
    ("median_inter_prompt_gap_s", "Median gap between prompts (s)"),
]
MANIPULATION_CHECKS = [
    ("time_to_first_tool_call_s", "Time to first tool call (s)",
     "slow_planning", "expected highest under **slow planning**"),
    ("time_to_first_prompt_s", "Time to first prompt (s)",
     "slow_planning", "expected highest under **slow planning** (history-file proxy)"),
    ("median_inter_tool_gap_s", "Median gap between tool calls (s)",
     "slow_iterating", "expected highest under **slow iterating**"),
    ("median_inter_prompt_gap_s", "Median gap between prompts (s)",
     "slow_iterating", "expected highest under **slow iterating** (history-file proxy)"),
]


# ---------------------------------------------------------------------------
# One-way ANOVA (no SciPy)
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def one_way_anova(groups: list[list[float]]) -> dict | None:
    """F, p, and eta-squared for >=2 groups with >=2 total observations and
    non-zero between/within structure. None if not computable."""
    groups = [[v for v in g if v is not None and not math.isnan(v)] for g in groups]
    groups = [g for g in groups if g]
    if len(groups) < 2:
        return None
    n = sum(len(g) for g in groups)
    k = len(groups)
    if n <= k:
        return None
    grand = sum(sum(g) for g in groups) / n
    ss_between = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups)
    ss_within = sum(sum((v - sum(g) / len(g)) ** 2 for v in g) for g in groups)
    ss_total = ss_between + ss_within
    df_b, df_w = k - 1, n - k
    if ss_within <= 0 or df_w <= 0:
        return None
    f_stat = (ss_between / df_b) / (ss_within / df_w)
    p = _betai(df_w / 2.0, df_b / 2.0, df_w / (df_w + df_b * f_stat))
    return {
        "F": f_stat, "p": p,
        "eta_sq": (ss_between / ss_total) if ss_total > 0 else 0.0,
        "df_between": df_b, "df_within": df_w,
    }


# ---------------------------------------------------------------------------
# Dataframe + charts
# ---------------------------------------------------------------------------

def build_cohort_df(summaries: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(summaries)
    if df.empty:
        return df
    df = df[df["status"] == "complete"].copy()
    df["condition_label"] = df["condition"].map(CONDITION_LABEL).fillna(df["condition"])
    return df


def _by_condition_figure(df: pd.DataFrame, col: str, label: str) -> go.Figure | None:
    sub = df[["condition", "condition_label", col]].dropna(subset=[col])
    if sub.empty:
        return None
    order = [c for c in CONDITIONS if c in sub["condition"].unique()]
    fig = px.box(
        sub, x="condition", y=col, category_orders={"condition": order},
        points="all", color="condition",
        color_discrete_map=CONDITION_COLOR,
        labels={col: label, "condition": ""},
    )
    fig.update_layout(
        showlegend=False, height=320, margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(tickvals=order, ticktext=[CONDITION_LABEL.get(c, c) for c in order]),
    )
    return fig


def _sec_label(section: str) -> str:
    return section.replace("_", " ").strip().capitalize()


def _section_order(rows: list[dict]) -> list[str]:
    seen = []
    for r in rows:
        if r["section"] not in seen:
            seen.append(r["section"])
    return seen


def render_notebook_by_section(section_rows: list[dict]) -> None:
    """Cohort notebook assessment broken out by rubric section, split by condition
    (each point is one participant's % of that section's max)."""
    rows = [r for r in section_rows if r.get("pct") is not None and r.get("condition")]
    if not rows:
        st.caption("No graded notebooks with a condition yet.")
        return
    df = pd.DataFrame(rows)
    df["section_label"] = df["section"].map(_sec_label)
    df["condition_label"] = df["condition"].map(CONDITION_LABEL).fillna(df["condition"])
    order = [_sec_label(s) for s in _section_order(rows)]
    cond_order = [c for c in CONDITIONS if c in df["condition"].unique()]

    fig = px.box(
        df, x="section_label", y="pct", color="condition",
        category_orders={"section_label": order, "condition": cond_order},
        color_discrete_map=CONDITION_COLOR, points="all",
        labels={"pct": "% of section max", "section_label": ""},
        hover_data=["subject_id"],
    )
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=20, b=10), boxmode="group",
        legend=dict(title="", orientation="h", y=1.12),
    )
    st.plotly_chart(fig, width="stretch")

    piv = (
        df.groupby(["section_label", "condition_label"])["pct"].mean()
        .round(1).unstack("condition_label")
        .reindex(order)
    )
    st.dataframe(piv, width="stretch")


def render_participant_notebook_sections(
    nb_results: list[dict], section_rows: list[dict], subject_id: str
) -> None:
    """One participant's per-section notebook score vs. the cohort mean per section."""
    by_sec: dict[str, list[float]] = {}
    for it in nb_results or []:
        s = (it.get("section") or "General").strip() or "General"
        got = by_sec.setdefault(s, [0.0, 0.0])
        got[0] += float(it.get("score") or 0)
        got[1] += float(it.get("max_pts") or 0)
    if not by_sec:
        return
    sections = list(by_sec)
    mine = {s: (100 * sc / mx if mx else 0.0) for s, (sc, mx) in by_sec.items()}

    cohort_pct: dict[str, list[float]] = {}
    for r in section_rows:
        if r.get("pct") is not None and r["subject_id"] != subject_id:
            cohort_pct.setdefault(r["section"], []).append(r["pct"])
    cohort_mean = {s: (sum(v) / len(v) if v else None) for s, v in cohort_pct.items()}

    labels = [_sec_label(s) for s in sections]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=[round(mine[s], 1) for s in sections],
        name=subject_id, marker_color=CONDITION_COLOR["control"],
    ))
    cm = [cohort_mean.get(s) for s in sections]
    if any(v is not None for v in cm):
        fig.add_trace(go.Scatter(
            x=labels, y=[round(v, 1) if v is not None else None for v in cm],
            name="cohort mean", mode="markers",
            marker=dict(symbol="diamond", size=12, color="#0b0b0b",
                        line=dict(width=2, color="#fcfcfb")),
        ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(title="% of section max", range=[0, 105]),
        legend=dict(orientation="h", y=1.15),
    )
    st.plotly_chart(fig, width="stretch")


def _stats_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for cond in CONDITIONS:
        vals = df.loc[df["condition"] == cond, col].dropna()
        rows.append({
            "condition": CONDITION_LABEL[cond],
            "n": int(vals.count()),
            "mean": round(vals.mean(), 2) if len(vals) else None,
            "sd": round(vals.std(ddof=1), 2) if len(vals) > 1 else None,
        })
    return pd.DataFrame(rows)


def render_cohort(summaries: list[dict], section_rows: list[dict] | None = None) -> None:
    df = build_cohort_df(summaries)
    if df.empty:
        st.info("No completed submissions yet.")
        return

    n_by_cond = df["condition"].value_counts().to_dict()
    st.markdown(
        "**Completed submissions:** "
        + " · ".join(
            f"{CONDITION_LABEL.get(c, c)} = {n_by_cond.get(c, 0)}" for c in CONDITIONS
        )
        + f" · total {len(df)}"
    )
    st.caption(
        "All comparisons below are **descriptive** — with this sample size, read the "
        "ANOVA F/p as a rough signal, not a hypothesis test."
    )

    st.markdown("### Outcomes")
    for col, label in OUTCOME_METRICS:
        if col not in df or df[col].dropna().empty:
            continue
        st.markdown(f"**{label}**")
        c1, c2 = st.columns([2, 1])
        fig = _by_condition_figure(df, col, label)
        if fig is not None:
            c1.plotly_chart(fig, width="stretch")
        with c2:
            st.dataframe(_stats_table(df, col), hide_index=True, width="stretch")
            aov = one_way_anova([df.loc[df["condition"] == c, col].dropna().tolist()
                                 for c in CONDITIONS])
            if aov:
                st.caption(
                    f"ANOVA F({aov['df_between']},{aov['df_within']}) = {aov['F']:.2f}, "
                    f"p = {aov['p']:.3f}, η² = {aov['eta_sq']:.2f}"
                )

    if section_rows:
        st.markdown("### Notebook — by rubric section")
        st.caption("Each point is one participant's score on that section, as % of "
                   "the section's max.")
        render_notebook_by_section(section_rows)

    st.markdown("### Behaviour")
    for col, label in BEHAVIOUR_METRICS:
        if col not in df or df[col].dropna().empty:
            continue
        fig = _by_condition_figure(df, col, label)
        if fig is not None:
            st.markdown(f"**{label}**")
            st.plotly_chart(fig, width="stretch")

    st.markdown("### Manipulation checks")
    st.caption("Did each intervention actually change the behaviour it targets?")
    for col, label, target, note in MANIPULATION_CHECKS:
        if col not in df or df[col].dropna().empty:
            continue
        tbl = _stats_table(df, col)
        means = {r["condition"]: r["mean"] for _, r in tbl.iterrows()}
        target_label = CONDITION_LABEL[target]
        highest = max((m for m in means.values() if m is not None), default=None)
        ok = highest is not None and means.get(target_label) == highest
        st.markdown(f"{'✅' if ok else '⚠️'} **{label}** — {note}")
        st.dataframe(tbl, hide_index=True, width="stretch")

    st.markdown("### Quiz vs. notebook score")
    sc = df.dropna(subset=["quiz_pct", "notebook_pct"])
    if len(sc) >= 2:
        fig = px.scatter(
            sc, x="notebook_pct", y="quiz_pct", color="condition",
            color_discrete_map=CONDITION_COLOR, hover_name="subject_id",
            labels={"notebook_pct": "Notebook score (%)", "quiz_pct": "Quiz score (%)"},
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("Need at least 2 participants with both scores.")
