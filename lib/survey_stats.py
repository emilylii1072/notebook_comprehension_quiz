"""Cohort-level survey visualisations for the admin.

Consumes `db.list_participant_survey_rows()` (one row per participant x survey
item) and renders the pre/post surveys the way the study wants to read them:
knowledge gain, attitude shift, workload, and self-assessment against the
actual notebook grade. Individual responses stay in
`lib.survey_ui.render_survey_breakdown` (Participant tab); this module is only
the across-participant view.

Colour follows the job, per the project's existing palette (lib/cohort.py):
  - condition  -> categorical, CONDITION_COLOR, fixed order, never cycled
  - pre vs post -> categorical, PRE_COLOR / POST_COLOR (validated pair)
  - agreement  -> diverging, warm..neutral grey..cool (polarity)
  - frequency / extent -> sequential single hue, light..dark (magnitude)
Every chart ships a data table next to or under it, which is also what relieves
the sub-3:1 contrast warning on the green.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib.cohort import (
    CONDITION_COLOR,
    CONDITION_LABEL,
    CONDITIONS,
    one_way_anova,
)
from lib.surveys import (
    ITEMS_BY_ID,
    KNOWLEDGE_PAIRS,
    MATCHED_LIKERT,
    POST_SURVEY_ITEMS,
    PRE_SURVEY_ITEMS,
    is_correct,
    scale_for,
)

PRE_COLOR, POST_COLOR = "#7b61c4", "#1baf7a"
PHASE_COLOR = {"Pre": PRE_COLOR, "Post": POST_COLOR}

# Same colours as CONDITION_COLOR, keyed by the human label — for charts that
# show a legend, so it reads "Slow planning" rather than "slow_planning".
CONDITION_COLOR_BY_LABEL = {
    CONDITION_LABEL[c]: CONDITION_COLOR[c] for c in CONDITIONS
}

# diverging anchors (warm negative -> neutral grey -> cool positive) and the
# sequential hue, both stepped to whatever length a scale needs.
_DIVERGING_ANCHORS = [(184, 67, 44), (214, 211, 205), (27, 175, 122)]
_SEQUENTIAL_ANCHORS = [(214, 228, 245), (21, 76, 138)]
_HEATMAP_ANCHORS = [(109, 152, 204), (16, 58, 107)]


def _ramp(anchors: list[tuple[int, int, int]], n: int) -> list[str]:
    """n evenly spaced colours along a piecewise-linear RGB ramp."""
    if n <= 1:
        return ["#%02x%02x%02x" % anchors[-1]]
    out = []
    span = len(anchors) - 1
    for i in range(n):
        pos = (i / (n - 1)) * span
        lo = min(int(pos), span - 1)
        t = pos - lo
        a, b = anchors[lo], anchors[lo + 1]
        out.append("#%02x%02x%02x" % tuple(round(a[c] + (b[c] - a[c]) * t) for c in range(3)))
    return out


def scale_colors(item_id: str, n: int) -> list[str]:
    """Diverging for agreement/likelihood (polarity), sequential otherwise."""
    item = ITEMS_BY_ID.get(item_id)
    opts = " ".join(item.options or []).lower() if item else ""
    polar = "agree" in opts or "likely" in opts
    return _ramp(_DIVERGING_ANCHORS if polar else _SEQUENTIAL_ANCHORS, n)


# ---------------------------------------------------------------------------
# Reshaping
# ---------------------------------------------------------------------------

_MATRIX_COLS = ["subject_id", "condition", "survey_type", "item_id",
                "statement", "value", "code"]
_SELECT_COLS = ["subject_id", "condition", "survey_type", "item_id",
                "option", "other_text"]


def _frame(records: list[dict], columns: list[str]) -> pd.DataFrame:
    """A DataFrame that still has its columns when there are no records, so
    callers can filter on them without a KeyError."""
    return pd.DataFrame(records, columns=columns)


def _is_matrix_answer(answer) -> bool:
    return isinstance(answer, dict) and "selected" not in answer


def matrix_long(rows: list[dict]) -> pd.DataFrame:
    """One row per (participant, matrix item, statement): the chosen scale point
    plus its 1..N code on the item's canonical (negative-first) ordering."""
    out = []
    for r in rows:
        if not _is_matrix_answer(r["answer"]):
            continue
        item = ITEMS_BY_ID.get(r["item_id"])
        if item is None:
            continue
        scale = scale_for(item) or (item.options or [])
        for statement, value in (r["answer"] or {}).items():
            out.append({
                "subject_id": r["subject_id"], "condition": r["condition"],
                "survey_type": r["survey_type"], "item_id": r["item_id"],
                "statement": statement, "value": value,
                "code": (scale.index(value) + 1) if value in scale else None,
            })
    return _frame(out, _MATRIX_COLS)


def select_long(rows: list[dict]) -> pd.DataFrame:
    """One row per (participant, select item, chosen option) — multi_select
    contributes one row per checked box."""
    out = []
    for r in rows:
        a = r["answer"]
        if not isinstance(a, dict) or "selected" not in a:
            continue
        sel = a.get("selected")
        chosen = sel if isinstance(sel, list) else ([sel] if sel else [])
        for opt in chosen:
            out.append({
                "subject_id": r["subject_id"], "condition": r["condition"],
                "survey_type": r["survey_type"], "item_id": r["item_id"],
                "option": opt, "other_text": a.get("other_text"),
            })
    return _frame(out, _SELECT_COLS)


def knowledge_scores(rows: list[dict]) -> pd.DataFrame:
    """Per participant: pre score, post score, and the change, over the 10
    matched concepts. Only participants with at least one scored item appear."""
    pre_ids = {p for p, _, _ in KNOWLEDGE_PAIRS}
    post_ids = {q for _, q, _ in KNOWLEDGE_PAIRS}
    acc: dict[str, dict] = {}
    for r in rows:
        iid = r["item_id"]
        if iid not in pre_ids and iid not in post_ids:
            continue
        ok = is_correct(iid, r["answer"])
        rec = acc.setdefault(r["subject_id"], {
            "subject_id": r["subject_id"], "condition": r["condition"],
            "pre_correct": 0, "pre_n": 0, "post_correct": 0, "post_n": 0,
        })
        side = "pre" if iid in pre_ids else "post"
        rec[f"{side}_n"] += 1
        rec[f"{side}_correct"] += 1 if ok else 0
    df = pd.DataFrame(acc.values())
    if df.empty:
        return df
    n = len(KNOWLEDGE_PAIRS)
    df["pre_pct"] = (100 * df["pre_correct"] / n).where(df["pre_n"] > 0).round(1)
    df["post_pct"] = (100 * df["post_correct"] / n).where(df["post_n"] > 0).round(1)
    df["delta_pct"] = (df["post_pct"] - df["pre_pct"]).round(1)
    df["condition_label"] = df["condition"].map(CONDITION_LABEL).fillna(df["condition"])
    return df


def per_question_accuracy(rows: list[dict]) -> pd.DataFrame:
    """% correct on each matched concept, pre vs post."""
    by_item: dict[str, list[bool]] = {}
    for r in rows:
        ok = is_correct(r["item_id"], r["answer"])
        if ok is not None:
            by_item.setdefault(r["item_id"], []).append(ok)
    out = []
    for pre_id, post_id, label in KNOWLEDGE_PAIRS:
        for iid, phase in ((pre_id, "Pre"), (post_id, "Post")):
            vals = by_item.get(iid) or []
            if vals:
                out.append({"concept": label, "phase": phase,
                            "pct": round(100 * sum(vals) / len(vals), 1), "n": len(vals)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Shared chart pieces
# ---------------------------------------------------------------------------

def _likert_stack(df: pd.DataFrame, item_id: str) -> go.Figure | None:
    """Stacked % distribution, one bar per statement, ordered negative-first."""
    sub = df[(df["item_id"] == item_id) & df["value"].notna()]
    if sub.empty:
        return None
    item = ITEMS_BY_ID.get(item_id)
    scale = scale_for(item) if item else None
    scale = scale or sorted(sub["value"].unique())
    statements = [s for s in (item.rows or []) if s in set(sub["statement"])] if item else []
    statements = statements or list(dict.fromkeys(sub["statement"]))
    colors = scale_colors(item_id, len(scale))

    counts = sub.groupby(["statement", "value"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=statements, columns=scale, fill_value=0)
    pct = counts.div(counts.sum(axis=1).replace(0, pd.NA), axis=0) * 100

    fig = go.Figure()
    for point, color in zip(scale, colors):
        fig.add_trace(go.Bar(
            y=[_wrap(s) for s in statements], x=pct[point], name=point,
            orientation="h", marker=dict(color=color, line=dict(width=2, color="#fcfcfb")),
            customdata=counts[point],
            hovertemplate="%{y}<br>" + point + ": %{x:.0f}% (n=%{customdata})<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", height=112 + 62 * len(statements),
        margin=dict(l=10, r=10, t=66, b=30),
        xaxis=dict(title="", range=[0, 100], ticksuffix="%"),
        yaxis=dict(autorange="reversed"),
        # traceorder="normal" so the legend reads in the same order the segments
        # are stacked — Plotly reverses it by default on stacked bars, which puts
        # "strongly agree" first on a chart that starts at "strongly disagree".
        legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0,
                    traceorder="normal", title=""),
        bargap=0.35,
    )
    return fig


def _wrap(text: str, width: int = 46) -> str:
    """Soft-wrap a long statement so the y-axis stays readable."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "<br>".join(lines)


def _cond_order(df: pd.DataFrame) -> list[str]:
    return [c for c in CONDITIONS if c in set(df["condition"].dropna())]


def _mean_table(df: pd.DataFrame, value_col: str, by: str = "condition") -> pd.DataFrame:
    rows = []
    for cond in _cond_order(df):
        vals = df.loc[df[by] == cond, value_col].dropna()
        rows.append({
            "condition": CONDITION_LABEL.get(cond, cond), "n": int(vals.count()),
            "mean": round(vals.mean(), 2) if len(vals) else None,
            "sd": round(vals.std(ddof=1), 2) if len(vals) > 1 else None,
        })
    return pd.DataFrame(rows)


def _anova_caption(df: pd.DataFrame, col: str) -> None:
    aov = one_way_anova(
        [df.loc[df["condition"] == c, col].dropna().tolist() for c in CONDITIONS]
    )
    if aov:
        st.caption(
            f"ANOVA F({aov['df_between']},{aov['df_within']}) = {aov['F']:.2f}, "
            f"p = {aov['p']:.3f}, η² = {aov['eta_sq']:.2f}"
        )


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def render_coverage(rows: list[dict]) -> None:
    taken = {(r["subject_id"], r["survey_type"]) for r in rows}
    conds = {r["subject_id"]: r["condition"] for r in rows}
    pre = {s for s, t in taken if t == "pre"}
    post = {s for s, t in taken if t == "post"}
    c1, c2, c3 = st.columns(3)
    c1.metric("Pre-survey taken", len(pre))
    c2.metric("Post-survey taken", len(post))
    c3.metric("Both", len(pre & post))

    tbl = pd.DataFrame([
        {
            "condition": CONDITION_LABEL.get(c, c),
            "pre": sum(1 for s in pre if conds.get(s) == c),
            "post": sum(1 for s in post if conds.get(s) == c),
            "both": sum(1 for s in (pre & post) if conds.get(s) == c),
        }
        for c in CONDITIONS
    ])
    st.dataframe(tbl, hide_index=True, width="stretch")

    st.markdown("**Time spent**")
    elapsed = pd.DataFrame([
        {"subject_id": r["subject_id"], "survey_type": r["survey_type"],
         "elapsed_seconds": r["elapsed_seconds"]}
        for r in rows
    ]).drop_duplicates(subset=["subject_id", "survey_type"]).dropna(subset=["elapsed_seconds"])
    if elapsed.empty:
        st.caption("No timing recorded yet.")
        return
    elapsed["phase"] = elapsed["survey_type"].map({"pre": "Pre", "post": "Post"})
    elapsed["minutes"] = elapsed["elapsed_seconds"].astype(float) / 60
    fig = px.box(
        elapsed, x="phase", y="minutes", color="phase", points="all",
        category_orders={"phase": ["Pre", "Post"]}, color_discrete_map=PHASE_COLOR,
        hover_name="subject_id", labels={"minutes": "Minutes", "phase": ""},
    )
    fig.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Slowest items** (median seconds on the question)")
    per_item = pd.DataFrame([
        {"item_id": r["item_id"], "survey_type": r["survey_type"],
         "seconds": r["time_spent_seconds"]}
        for r in rows if r["time_spent_seconds"] is not None
    ])
    if per_item.empty:
        st.caption("No per-item timing recorded yet.")
        return
    med = (per_item.groupby(["survey_type", "item_id"])["seconds"].median()
           .reset_index().sort_values("seconds", ascending=False).head(15))
    med["label"] = med["item_id"] + "  (" + med["survey_type"] + ")"
    med["phase"] = med["survey_type"].map({"pre": "Pre", "post": "Post"})
    fig = px.bar(
        med.sort_values("seconds"), x="seconds", y="label", orientation="h",
        color="phase", color_discrete_map=PHASE_COLOR,
        labels={"seconds": "Median seconds", "label": ""},
    )
    fig.update_traces(marker_line=dict(width=2, color="#fcfcfb"))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=20, b=10),
                      legend=dict(orientation="h", y=1.08, title=""))
    st.plotly_chart(fig, width="stretch")


def render_knowledge(rows: list[dict]) -> None:
    st.caption(
        "Scored against `lib.surveys.ANSWER_KEY` — a key written for this app, not "
        "taken from the Qualtrics export. Check it before reporting a learning gain."
    )
    scores = knowledge_scores(rows)
    if scores.empty:
        st.info("No knowledge-assessment answers yet.")
        return

    both = scores.dropna(subset=["pre_pct", "post_pct"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean pre score", f"{scores['pre_pct'].mean():.0f}%"
              if scores["pre_pct"].notna().any() else "—")
    c2.metric("Mean post score", f"{scores['post_pct'].mean():.0f}%"
              if scores["post_pct"].notna().any() else "—")
    c3.metric("Mean change", f"{both['delta_pct'].mean():+.0f} pts" if len(both) else "—",
              help="Post minus pre, over participants who took both.")

    if len(both):
        st.markdown("**Each participant, pre → post**")
        fig = go.Figure()
        for _, r in both.iterrows():
            color = CONDITION_COLOR.get(r["condition"], "#8b8b8b")
            fig.add_trace(go.Scatter(
                x=["Pre", "Post"], y=[r["pre_pct"], r["post_pct"]],
                mode="lines+markers", name=r["subject_id"],
                line=dict(color=color, width=2),
                marker=dict(size=9, color=color, line=dict(width=2, color="#fcfcfb")),
                hovertemplate=f"{r['subject_id']} ({r['condition_label']})<br>%{{x}}: %{{y:.0f}}%<extra></extra>",
                showlegend=False,
            ))
        for cond in _cond_order(both):
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines", name=CONDITION_LABEL.get(cond, cond),
                line=dict(color=CONDITION_COLOR.get(cond), width=2),
            ))
        fig.update_layout(
            height=400, margin=dict(l=10, r=10, t=20, b=10),
            yaxis=dict(title="% correct", range=[-5, 105]),
            xaxis=dict(title=""), legend=dict(orientation="h", y=1.12, title=""),
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown("**Change by condition**")
        c1, c2 = st.columns([2, 1])
        fig = px.box(
            both, x="condition", y="delta_pct", color="condition", points="all",
            category_orders={"condition": _cond_order(both)},
            color_discrete_map=CONDITION_COLOR, hover_name="subject_id",
            labels={"delta_pct": "Change (pts)", "condition": ""},
        )
        order = _cond_order(both)
        fig.update_layout(
            showlegend=False, height=340, margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(tickvals=order, ticktext=[CONDITION_LABEL.get(c, c) for c in order]),
        )
        fig.add_hline(y=0, line_dash="dot", line_color="#9a9a9a")
        c1.plotly_chart(fig, width="stretch")
        with c2:
            st.dataframe(_mean_table(both, "delta_pct"), hide_index=True, width="stretch")
            _anova_caption(both, "delta_pct")

    st.markdown("**Accuracy per concept**")
    acc = per_question_accuracy(rows)
    if acc.empty:
        st.caption("Nothing scored yet.")
    else:
        order = [label for _, _, label in KNOWLEDGE_PAIRS if label in set(acc["concept"])]
        fig = px.bar(
            acc, x="pct", y="concept", color="phase", orientation="h", barmode="group",
            category_orders={"concept": order[::-1], "phase": ["Pre", "Post"]},
            color_discrete_map=PHASE_COLOR, custom_data=["n"],
            labels={"pct": "% correct", "concept": ""},
        )
        fig.update_traces(marker_line=dict(width=2, color="#fcfcfb"),
                          hovertemplate="%{y}<br>%{x:.0f}% correct (n=%{customdata[0]})<extra></extra>")
        fig.update_layout(height=80 + 46 * len(order), margin=dict(l=10, r=10, t=20, b=10),
                          xaxis=dict(range=[0, 100], ticksuffix="%"),
                          legend=dict(orientation="h", y=1.06, title=""))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(
            acc.pivot(index="concept", columns="phase", values="pct").reindex(order),
            width="stretch",
        )

    with st.expander("Per-participant scores"):
        st.dataframe(
            scores[["subject_id", "condition_label", "pre_pct", "post_pct", "delta_pct"]]
            .sort_values("subject_id"),
            hide_index=True, width="stretch",
        )


def render_attitudes(rows: list[dict]) -> None:
    mat = matrix_long(rows)
    if mat.empty:
        st.info("No matrix answers yet.")
        return

    st.markdown("### Shift on the statements asked twice")
    shift_rows = []
    for pre_item, pre_stmt, post_item, post_stmt, label in MATCHED_LIKERT:
        pre_vals = mat[(mat["item_id"] == pre_item) & (mat["statement"] == pre_stmt)]
        post_vals = mat[(mat["item_id"] == post_item) & (mat["statement"] == post_stmt)]
        merged = pre_vals.merge(post_vals, on="subject_id", suffixes=("_pre", "_post"))
        for _, r in merged.iterrows():
            if pd.notna(r["code_pre"]) and pd.notna(r["code_post"]):
                shift_rows.append({
                    "statement": label, "subject_id": r["subject_id"],
                    "condition": r["condition_pre"],
                    "pre": r["code_pre"], "post": r["code_post"],
                    "shift": r["code_post"] - r["code_pre"],
                })
    if shift_rows:
        sh = pd.DataFrame(shift_rows)
        agg = sh.groupby("statement")[["pre", "post"]].mean().reset_index()
        long = agg.melt(id_vars="statement", var_name="phase", value_name="mean")
        long["phase"] = long["phase"].map({"pre": "Pre", "post": "Post"})
        fig = px.bar(
            long, x="mean", y="statement", color="phase", orientation="h", barmode="group",
            category_orders={"phase": ["Pre", "Post"],
                             "statement": [m[4] for m in MATCHED_LIKERT]},
            color_discrete_map=PHASE_COLOR,
            labels={"mean": "Mean (1 = strongly disagree … 5 = strongly agree)", "statement": ""},
        )
        fig.update_traces(marker_line=dict(width=2, color="#fcfcfb"))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10),
                          xaxis=dict(range=[1, 5]),
                          legend=dict(orientation="h", y=1.15, title=""))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(
            sh.groupby("statement")[["pre", "post", "shift"]].mean().round(2)
            .reindex([m[4] for m in MATCHED_LIKERT]).dropna(how="all"),
            width="stretch",
        )
    else:
        st.caption("Need participants with both surveys to show the shift.")

    st.markdown("### Response distributions")
    matrix_items = [
        (it.id, it) for it in (*PRE_SURVEY_ITEMS, *POST_SURVEY_ITEMS)
        if it.kind == "matrix" and not it.id.startswith("workload_")
    ]
    present = [(iid, it) for iid, it in matrix_items if iid in set(mat["item_id"])]
    for iid, it in present:
        phase = "Pre" if any(x.id == iid for x in PRE_SURVEY_ITEMS) else "Post"
        fig = _likert_stack(mat, iid)
        if fig is not None:
            st.markdown(f"**{phase} · {it.question}**")
            st.plotly_chart(fig, width="stretch")

    st.markdown("### Role of AI agents — before vs after")
    sel = select_long(rows)
    role = sel[sel["item_id"].isin(["ai_collab_role", "post_ai_role"])]
    if role.empty:
        st.caption("No answers yet.")
        return
    role = role.assign(phase=role["item_id"].map({"ai_collab_role": "Pre", "post_ai_role": "Post"}))
    counts = role.groupby(["option", "phase"]).size().reset_index(name="n")
    opts = ITEMS_BY_ID["ai_collab_role"].options or []
    counts["short"] = counts["option"].str.split(" — ").str[0]
    order = [o.split(" — ")[0] for o in opts]
    fig = px.bar(
        counts, x="n", y="short", color="phase", orientation="h", barmode="group",
        category_orders={"short": order[::-1], "phase": ["Pre", "Post"]},
        color_discrete_map=PHASE_COLOR, hover_data=["option"],
        labels={"n": "Participants", "short": ""},
    )
    fig.update_traces(marker_line=dict(width=2, color="#fcfcfb"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10),
                      legend=dict(orientation="h", y=1.12, title=""))
    st.plotly_chart(fig, width="stretch")


def render_workload(rows: list[dict]) -> None:
    mat = matrix_long(rows)
    wl = mat[mat["item_id"].str.startswith("workload_")] if not mat.empty else mat
    if wl.empty:
        st.info("No workload answers yet.")
        return
    labels = {
        "workload_main_task": "Main task (notebook)",
        "workload_idea_gen": "Idea generation",
        "workload_debugging": "Debugging",
        "workload_interview": "Interview",
    }
    short = {
        "How mentally demanding or difficult was this task?": "Mental demand",
        "How hard/effortful did you have to work to complete this task?": "Effort",
        "How hurried or rushed was the pace of the task?": "Time pressure",
        "How familiar are you with the type of task?": "Familiarity",
    }

    def _short(s: str) -> str:
        if s in short:
            return short[s]
        return "Frustration" if "irritated" in s else s

    wl = wl.assign(task=wl["item_id"].map(labels), dimension=wl["statement"].map(_short))
    st.caption("1 = not at all · 5 = extremely. Each cell is the cohort mean.")

    piv = wl.pivot_table(index="dimension", columns="task", values="code", aggfunc="mean")
    dim_order = ["Mental demand", "Effort", "Frustration", "Time pressure", "Familiarity"]
    task_order = [t for t in labels.values() if t in piv.columns]
    piv = piv.reindex(index=[d for d in dim_order if d in piv.index], columns=task_order)
    fig = px.imshow(
        piv, color_continuous_scale=_ramp(_HEATMAP_ANCHORS, 9), zmin=1, zmax=5,
        text_auto=".1f", aspect="auto", labels=dict(color="Mean"),
    )
    # The ramp starts mid-blue rather than near-white so one forced text colour
    # stays legible on every cell; exact values are in the table underneath.
    fig.update_traces(textfont=dict(color="#ffffff", size=13))
    fig.update_xaxes(side="top", title="")
    fig.update_yaxes(title="")
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(piv.round(2), width="stretch")

    st.markdown("**By condition**")
    task_pick = st.selectbox("Task", task_order, key="wl_task")
    sub = wl[wl["task"] == task_pick].dropna(subset=["code"])
    if sub.empty:
        st.caption("Nothing recorded for that task yet.")
        return
    means = (sub.groupby(["dimension", "condition"])["code"].mean().reset_index())
    means["condition_label"] = means["condition"].map(CONDITION_LABEL).fillna(means["condition"])
    fig = px.bar(
        means, x="dimension", y="code", color="condition_label", barmode="group",
        category_orders={"dimension": [d for d in dim_order if d in set(means["dimension"])],
                         "condition_label": [CONDITION_LABEL[c] for c in _cond_order(means)]},
        color_discrete_map=CONDITION_COLOR_BY_LABEL,
        labels={"code": "Mean (1–5)", "dimension": ""},
    )
    fig.update_traces(marker_line=dict(width=2, color="#fcfcfb"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10),
                      yaxis=dict(range=[0, 5]), legend=dict(orientation="h", y=1.12, title=""))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        means.pivot(index="dimension", columns="condition_label", values="code").round(2),
        width="stretch",
    )


def render_self_assessment(rows: list[dict], summaries: list[dict]) -> None:
    grades = {}
    for r in rows:
        if r["item_id"] in ("post_grade_validity", "post_grade_completeness"):
            grades.setdefault(r["subject_id"], {})[r["item_id"]] = r["answer"]
    if not grades:
        st.info("No self-assessed grades yet.")
        return
    actual = {s["subject_id"]: s for s in summaries}
    df = pd.DataFrame([
        {
            "subject_id": sid,
            "condition": actual.get(sid, {}).get("condition"),
            "self_validity": pd.to_numeric(g.get("post_grade_validity"), errors="coerce"),
            "self_completeness": pd.to_numeric(g.get("post_grade_completeness"), errors="coerce"),
            "notebook_pct": actual.get(sid, {}).get("notebook_pct"),
        }
        for sid, g in grades.items()
    ])
    df["self_mean_pct"] = (df[["self_validity", "self_completeness"]].mean(axis=1) / 4 * 100).round(1)
    df["condition_label"] = df["condition"].map(CONDITION_LABEL).fillna(df["condition"])

    st.caption("Participants graded their own notebook 0–4 on validity and completeness. "
               "Plotted against the rubric grade the app computed.")
    sc = df.dropna(subset=["self_mean_pct", "notebook_pct"])
    if len(sc) >= 2:
        fig = px.scatter(
            sc, x="notebook_pct", y="self_mean_pct", color="condition_label",
            color_discrete_map=CONDITION_COLOR_BY_LABEL, hover_name="subject_id",
            category_orders={"condition_label": [CONDITION_LABEL[c] for c in _cond_order(sc)]},
            labels={"notebook_pct": "Rubric grade (%)", "self_mean_pct": "Self-assessed (%)"},
        )
        fig.update_traces(marker=dict(size=12, line=dict(width=2, color="#fcfcfb")))
        fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                      line=dict(color="#9a9a9a", dash="dot"))
        fig.add_annotation(x=82, y=92, text="over-estimates", showarrow=False,
                           font=dict(size=11, color="#6b6b6b"))
        fig.add_annotation(x=92, y=12, text="under-estimates", showarrow=False,
                           font=dict(size=11, color="#6b6b6b"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10),
                          xaxis=dict(range=[0, 105]), yaxis=dict(range=[0, 105]),
                          legend=dict(orientation="h", y=1.12, title=""))
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("Need at least 2 participants with both a self-grade and a rubric grade.")
    st.dataframe(
        df[["subject_id", "self_validity", "self_completeness", "self_mean_pct", "notebook_pct"]]
        .sort_values("subject_id"), hide_index=True, width="stretch",
    )

    st.markdown("**What the AI did for the task**")
    sel = select_long(rows)
    parts = sel[sel["item_id"] == "post_ai_parts"]
    if parts.empty:
        st.caption("No answers yet.")
        return
    n_resp = parts["subject_id"].nunique()
    counts = parts.groupby("option").size().reset_index(name="n")
    counts["pct"] = (100 * counts["n"] / n_resp).round(1)
    opts = ITEMS_BY_ID["post_ai_parts"].options or []
    order = [o for o in opts if o in set(counts["option"])]
    fig = px.bar(
        counts, x="pct", y="option", orientation="h",
        category_orders={"option": order[::-1]}, custom_data=["n"],
        labels={"pct": "% of participants", "option": ""},
    )
    fig.update_traces(marker=dict(color="#2a78d6", line=dict(width=2, color="#fcfcfb")),
                      hovertemplate="%{y}<br>%{x:.0f}% (n=%{customdata[0]})<extra></extra>")
    fig.update_layout(height=80 + 42 * len(order), margin=dict(l=10, r=10, t=20, b=10),
                      xaxis=dict(range=[0, 100], ticksuffix="%"))
    st.plotly_chart(fig, width="stretch")


def render_background(rows: list[dict]) -> None:
    """Demographics and prior experience — the pre-survey's select items."""
    sel = select_long(rows)
    if sel.empty:
        st.info("No answers yet.")
        return
    wanted = [
        ("logistics_gender", "Gender"),
        ("logistics_first_language", "First language"),
        ("logistics_occupation", "Occupation"),
        ("prior_experience_ds_level", "Data-science experience"),
        ("ai_experience_interactions", "Ways they have interacted with AI"),
        ("ai_collab_time_pref", "Preferred way to spend time with AI"),
    ]
    present = [(iid, lab) for iid, lab in wanted if iid in set(sel["item_id"])]
    if not present:
        st.caption("No background answers yet.")
    for iid, label in present:
        sub = sel[sel["item_id"] == iid]
        counts = sub.groupby("option").size().reset_index(name="n")
        opts = (ITEMS_BY_ID[iid].options or [])
        order = [o for o in opts if o in set(counts["option"])]
        fig = px.bar(
            counts, x="n", y="option", orientation="h",
            category_orders={"option": order[::-1]},
            labels={"n": "Participants", "option": ""},
        )
        fig.update_traces(marker=dict(color="#2a78d6", line=dict(width=2, color="#fcfcfb")))
        fig.update_layout(height=70 + 38 * max(len(order), 1),
                          margin=dict(l=10, r=10, t=36, b=10),
                          title=dict(text=label, font=dict(size=14)),
                          xaxis=dict(dtick=1))
        st.plotly_chart(fig, width="stretch")
        free = sub["other_text"].dropna()
        free = [t for t in free if str(t).strip()]
        if free:
            st.caption("Free text: " + " · ".join(sorted(set(free))))

    ages = [r["answer"] for r in rows if r["item_id"] == "logistics_age"]
    ages = pd.to_numeric(pd.Series(ages), errors="coerce").dropna()
    if len(ages):
        st.markdown(f"**Age** — median {ages.median():.0f}, range {ages.min():.0f}–{ages.max():.0f}")


def render_free_text(rows: list[dict]) -> None:
    items = [
        ("post_ai_collab_strategy", "AI-collaboration strategy that would help"),
        ("post_demanding_part", "Particularly demanding / challenging"),
        ("post_other_comments", "Anything else"),
    ]
    any_shown = False
    for iid, label in items:
        answers = [
            (r["subject_id"], str(r["answer"]).strip())
            for r in rows if r["item_id"] == iid and str(r["answer"] or "").strip()
        ]
        if not answers:
            continue
        any_shown = True
        st.markdown(f"**{label}**")
        for sid, text in sorted(answers):
            st.markdown(f"- **{sid}** — {text}")
        st.divider()
    if not any_shown:
        st.info("No free-text answers yet.")


def render_survey_cohort(rows: list[dict], summaries: list[dict]) -> None:
    if not rows:
        st.info(
            "No survey responses yet. (Or the database isn't configured — see the "
            "Grading tab.)"
        )
        return
    n_people = len({r["subject_id"] for r in rows})
    st.markdown(f"**{n_people}** participant(s) have answered at least one survey.")
    st.caption(
        "Descriptive only — with this sample size read any F/p as a rough signal, "
        "not a hypothesis test. Individual responses live in the Participant tab."
    )

    t_cov, t_know, t_att, t_load, t_self, t_bg, t_txt, t_raw = st.tabs([
        "Coverage & timing", "Knowledge", "Attitudes", "Workload",
        "Self-assessment", "Background", "Free text", "Raw / export",
    ])
    with t_cov:
        render_coverage(rows)
    with t_know:
        render_knowledge(rows)
    with t_att:
        render_attitudes(rows)
    with t_load:
        render_workload(rows)
    with t_self:
        render_self_assessment(rows, summaries)
    with t_bg:
        render_background(rows)
    with t_txt:
        render_free_text(rows)
    with t_raw:
        flat = pd.DataFrame([
            {
                "subject_id": r["subject_id"], "condition": r["condition"],
                "survey_type": r["survey_type"], "item_id": r["item_id"],
                "category": r["category"], "question": r["question"],
                "answer": _answer_text(r["answer"]),
                "correct": is_correct(r["item_id"], r["answer"]),
                "time_spent_seconds": r["time_spent_seconds"],
            }
            for r in rows
        ])
        st.dataframe(flat, hide_index=True, width="stretch")
        st.download_button(
            "⬇️ Download survey responses (CSV)",
            data=flat.to_csv(index=False).encode("utf-8"),
            file_name="survey_responses.csv", mime="text/csv",
        )


def _answer_text(answer) -> str:
    """One flat cell per answer, for the export table."""
    if answer is None:
        return ""
    if isinstance(answer, dict) and "selected" in answer:
        sel = answer.get("selected")
        text = "; ".join(sel) if isinstance(sel, list) else (sel or "")
        other = answer.get("other_text")
        return f"{text} [{other}]" if other else text
    if isinstance(answer, dict):
        return "; ".join(f"{k} = {v}" for k, v in answer.items())
    return str(answer)
