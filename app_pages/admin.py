"""Admin — the researcher's view of every participant.

Password-gated (set ADMIN_PASSWORD in secrets / .env). Five tabs:
  1. Overview        — one row per participant, filter by condition, CSV export
  2. Participant     — the five docs, the graded notebook, the quiz, the verbal
                       assessment transcript, the session timeline
  3. Cohort stats    — outcomes + behaviour overall and split by condition
  4. Notebook report — the visual grading report across all graded notebooks
  5. Grading         — the active rubric + "grade all pending"

One page of the multipage app — run via `streamlit run app.py`.
"""

import io

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from build_report import records_from_graded, render_report
from db import (
    get_participant_bundle,
    get_rubric,
    get_secret,
    list_graded_participant_notebooks,
    list_notebook_section_scores,
    list_participant_logs_raw,
    list_participant_summaries,
    list_pending_grading,
    list_rubrics,
    save_participant_log,
    save_participant_transcript,
    save_rubric,
    set_participant_grading_error,
    supabase_status,
    update_participant_grading,
)
from lib import cohort, grading, transcript
from lib.llm import get_client
from lib.quiz_ui import render_quiz_breakdown
from lib.timeline import compute_log_metrics, parse_jsonl, render_timeline

load_dotenv()

CONDITION_LABEL = cohort.CONDITION_LABEL
ACTIVE_RUBRIC_NAME = get_secret("ACTIVE_RUBRIC_NAME", grading.DEFAULT_RUBRIC_NAME)
DOC_TITLES = {
    "task_plan": "Task plan",
    "debug_manual": "Debug — manual", "debug_ai": "Debug — AI",
    "ideate_manual": "Ideate — manual", "ideate_ai": "Ideate — AI",
}

st.title("🔐 Admin")

# ---- Gate -----------------------------------------------------------------
_expected_pw = get_secret("ADMIN_PASSWORD")
if not st.session_state.get("admin_ok"):
    if not _expected_pw:
        st.error("ADMIN_PASSWORD is not configured — set it in secrets / .env to use this page.")
        st.stop()
    pw = st.text_input("Admin password", type="password")
    if st.button("Enter"):
        if pw == _expected_pw:
            st.session_state.admin_ok = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


def _grade_one(subject_id: str, notebook_text: str) -> tuple[bool, str]:
    """Grade one participant's notebook against the active rubric. Returns (ok, msg)."""
    rubric = get_rubric(ACTIVE_RUBRIC_NAME)
    if rubric is None or not (rubric.get("rubric_csv") or "").strip():
        set_participant_grading_error(subject_id, f"No usable rubric '{ACTIVE_RUBRIC_NAME}'.")
        return False, f"No usable rubric '{ACTIVE_RUBRIC_NAME}'."
    try:
        results = grading.grade_notebook(
            get_client(), rubric["task"], rubric["rubric_csv"], notebook_text
        )
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        set_participant_grading_error(subject_id, msg)
        return False, msg
    total = round(sum(r["score"] for r in results), 2)
    mx = round(sum(r["max_pts"] for r in results), 2)
    ok, err = update_participant_grading(
        subject_id, ACTIVE_RUBRIC_NAME, grading.MODEL, results, total, mx
    )
    return (ok, "graded" if ok else (err or "save failed"))


tab_overview, tab_detail, tab_cohort, tab_report, tab_grading = st.tabs(
    ["1 · Overview", "2 · Participant", "3 · Cohort stats",
     "4 · Notebook report", "5 · Grading & rubric"]
)

_section_rows = list_notebook_section_scores()  # (participant, section) grades; used in tabs 2 & 3

# ---- Tab 1: Overview ----------------------------------------------------
with tab_overview:
    summaries = list_participant_summaries()
    if not summaries:
        st.info("No participants yet. (Or the database isn't configured — see the Grading tab.)")
    else:
        df = pd.DataFrame(summaries)
        conds = ["(all)"] + [c for c in cohort.CONDITIONS if c in df["condition"].unique()]
        pick = st.selectbox("Condition", conds, format_func=lambda c: CONDITION_LABEL.get(c, c))
        view = df if pick == "(all)" else df[df["condition"] == pick]

        show = view[[
            "subject_id", "condition", "status", "grading_status", "submitted_at",
            "notebook_pct", "quiz_score", "quiz_total", "session_duration_s",
            "n_tool_calls", "n_edits", "has_transcript",
        ]].copy()
        show["condition"] = show["condition"].map(CONDITION_LABEL).fillna(show["condition"])
        show["notebook_pct"] = show["notebook_pct"].round(1)
        st.dataframe(show, hide_index=True, width="stretch")
        st.download_button(
            "⬇️ Download all participants (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="participants.csv", mime="text/csv",
        )

# ---- Tab 2: Participant detail ----------------------------------------
with tab_detail:
    summaries = list_participant_summaries()
    ids = [s["subject_id"] for s in summaries]
    if not ids:
        st.info("No participants yet.")
    else:
        sid = st.selectbox("Subject", ids)
        bundle = get_participant_bundle(sid)
        p = bundle["participant"] or {}
        st.markdown(
            f"**{sid}** · {CONDITION_LABEL.get(p.get('condition'), p.get('condition'))} · "
            f"status `{p.get('status')}` · grading `{p.get('grading_status')}`"
        )
        if p.get("grading_status") == "error":
            st.warning(f"Grading error: {p.get('grading_error')}")

        docs = {d["doc_type"]: d for d in bundle["files"]}

        st.markdown("### Documents")
        if "task_plan" in docs:
            with st.expander("Task plan", expanded=False):
                st.markdown(docs["task_plan"]["content"])
        for left, right in (("debug_manual", "debug_ai"), ("ideate_manual", "ideate_ai")):
            c1, c2 = st.columns(2)
            for col, key in ((c1, left), (c2, right)):
                with col:
                    st.markdown(f"**{DOC_TITLES[key]}**")
                    st.markdown(docs[key]["content"] if key in docs else "_missing_")

        st.markdown("### Notebook grade")
        nb = bundle["notebook"]
        if nb and nb.get("results"):
            st.metric(
                "Total",
                f"{nb.get('total_score')} / {nb.get('max_score')}"
                + (f"  ({100 * nb['total_score'] / nb['max_score']:.0f}%)"
                   if nb.get("max_score") else ""),
            )
            cohort.render_participant_notebook_sections(nb["results"], _section_rows, sid)
            with st.expander("Per-criterion detail"):
                gdf = pd.DataFrame(nb["results"])[
                    ["section", "criterion", "score", "max_pts", "reasoning"]
                ]
                st.dataframe(gdf, hide_index=True, width="stretch")
        elif nb:
            st.caption("Notebook stored but not graded yet.")
            if st.button("Grade now", key="grade_now_detail"):
                with st.spinner("Grading…"):
                    ok, msg = _grade_one(sid, nb["notebook_text"])
                (st.success if ok else st.error)(msg)
                st.rerun()
        else:
            st.caption("No notebook on file.")

        if nb and nb.get("notebook_text"):
            with st.expander("Notebook transcript (as graded)"):
                st.code(nb["notebook_text"])

        st.markdown("### Comprehension quiz")
        if bundle["quiz"]:
            render_quiz_breakdown(bundle["quiz"])
        else:
            st.caption("No quiz result on file.")

        st.markdown("### Verbal assessment")
        tr = bundle["transcript"]
        if tr:
            pairs = tr.get("parsed") or []
            for pr in pairs:
                st.markdown(f"**`{pr.get('timestamp', '')}` · {pr.get('question', '')}**")
                st.markdown(pr.get("answer") or "_(no answer)_")
            if not pairs:
                st.caption("Uploaded but not parsed — use Re-parse below.")
            with st.expander("Raw transcript"):
                st.text(tr["raw_text"])
            c_re, c_dl = st.columns(2)
            if c_re.button("Re-parse", key="reparse_tr"):
                with st.spinner("Parsing…"):
                    parsed = transcript.call_with_errors_surfaced(
                        transcript.parse_transcript, get_client(), tr["raw_text"]
                    )
                if parsed is not None:
                    save_participant_transcript(sid, tr["filename"], tr["raw_text"], parsed)
                    st.rerun()
            c_dl.download_button(
                f"⬇️ {tr['filename']}", data=tr["raw_text"], file_name=tr["filename"],
                mime="text/plain", key="dl_tr",
            )
        else:
            up = st.file_uploader(
                "Verbal assessment transcript (.txt)", type=["txt"], key="tr_up"
            )
            if up is not None and st.button("Upload & parse", key="tr_parse", type="primary"):
                raw = up.getvalue().decode("utf-8", errors="replace")
                with st.spinner("Parsing the transcript…"):
                    parsed = transcript.call_with_errors_surfaced(
                        transcript.parse_transcript, get_client(), raw
                    )
                ok, err = save_participant_transcript(sid, up.name, raw, parsed)
                if not ok:
                    st.error(f"Could not save: {err}")
                else:
                    st.rerun()

        st.markdown("### Session timeline")
        log = bundle["log"]
        if log and log.get("raw_jsonl"):
            metrics = compute_log_metrics(parse_jsonl(log["raw_jsonl"]))  # recompute live
            mc = st.columns(4)
            dur = (metrics.get("session_duration_s") or 0) / 60
            mc[0].metric("Duration", f"{dur:.1f} min")
            if metrics.get("format") == "history":
                mc[1].metric("Prompts", metrics.get("n_substantive_prompts", 0))
                mc[2].metric("Sessions", metrics.get("n_sessions", 0))
                g = metrics.get("median_inter_prompt_gap_s")
                mc[3].metric("Median gap", f"{g:.0f}s" if g is not None else "—")
            else:
                mc[1].metric("Tool calls", metrics.get("n_tool_calls", 0))
                mc[2].metric("File edits", metrics.get("n_edits", 0))
                ttf = metrics.get("time_to_first_tool_call_s")
                mc[3].metric("To 1st tool", f"{ttf:.0f}s" if ttf is not None else "—")
            render_timeline(log["raw_jsonl"], key=sid)
        else:
            st.caption("No session log on file.")

        st.markdown("### Raw files")
        for d in bundle["files"]:
            st.download_button(
                f"⬇️ {d['filename']}", data=d["content"], file_name=d["filename"],
                mime="text/markdown", key=f"dl_{d['doc_type']}",
            )
        if nb:
            st.download_button(
                f"⬇️ {nb['filename']} (transcript .txt)", data=nb.get("notebook_text", ""),
                file_name=nb["filename"].replace(".ipynb", "_transcript.txt"),
                mime="text/plain", key="dl_nb",
            )
        if log:
            st.download_button(
                f"⬇️ {log['filename']}", data=log["raw_jsonl"],
                file_name=log["filename"], mime="application/json", key="dl_log",
            )

# ---- Tab 3: Cohort statistics ----------------------------------------
with tab_cohort:
    cohort.render_cohort(list_participant_summaries(), _section_rows)

# ---- Tab 4: Notebook grading report --------------------------------
with tab_report:
    st.markdown(
        "Visual review of every graded participant notebook: score distribution, "
        "per-section spread, a notebook × criterion heatmap, the criteria the cohort "
        "did worst on, and a clustering of scoring profiles."
    )
    graded = list_graded_participant_notebooks()
    if not graded:
        st.info("No graded notebooks yet.")
    else:
        records = records_from_graded(graded)
        report_html = render_report(records, ACTIVE_RUBRIC_NAME)
        st.download_button(
            "⬇️ Download report (single HTML file)",
            data=report_html, file_name=f"notebook_report_{ACTIVE_RUBRIC_NAME}.html",
            mime="text/html",
        )
        # Isolated iframe: the report ships its own CSS reset + tooltip script.
        components.html(report_html, height=2200, scrolling=True)

# ---- Tab 5: Grading & rubric ----------------------------------------
with tab_grading:
    st.markdown(
        f"Hidden-synchronous grading uses the rubric named **`{ACTIVE_RUBRIC_NAME}`** "
        "(override with the `ACTIVE_RUBRIC_NAME` secret)."
    )
    active = get_rubric(ACTIVE_RUBRIC_NAME)
    if active and (active.get("rubric_csv") or "").strip():
        st.success(f"Rubric `{ACTIVE_RUBRIC_NAME}` is configured.")
    else:
        st.error(
            f"No usable rubric `{ACTIVE_RUBRIC_NAME}` — new submissions will save but "
            "stay ungraded until you upload one below and re-grade them."
        )

    st.markdown("#### Upload / replace a rubric")
    existing = list_rubrics()
    if existing:
        st.caption("Saved rubrics: " + ", ".join(f"`{r}`" for r in existing))
    name = st.text_input("Rubric name", value=ACTIVE_RUBRIC_NAME)
    task_text = st.text_area("Task description (graded against)", value=grading.DEFAULT_TASK, height=220)
    up = st.file_uploader("Rubric CSV", type=["csv"])
    if up is not None and st.button("Save rubric", type="primary"):
        try:
            csv_text = up.getvalue().decode("utf-8-sig")
        except UnicodeDecodeError:
            csv_text = up.getvalue().decode("latin-1")
        ok, err = save_rubric(name.strip() or ACTIVE_RUBRIC_NAME, task_text, csv_text)
        (st.success if ok else st.error)("Saved." if ok else f"Not saved: {err}")
        st.rerun()

    if active and (active.get("rubric_csv") or "").strip():
        try:
            st.dataframe(pd.read_csv(io.StringIO(active["rubric_csv"])), hide_index=True, width="stretch")
        except Exception:
            st.code(active["rubric_csv"], language="csv")

    st.divider()
    st.markdown("#### Grade pending participants")
    pending = list_pending_grading()
    st.caption(f"{len(pending)} participant notebook(s) pending or errored.")
    if pending and st.button(f"Grade {len(pending)} notebook(s)", type="primary"):
        prog = st.progress(0.0, text="Starting…")
        for i, row in enumerate(pending):
            prog.progress(i / len(pending), text=f"Grading {row['subject_id']}…")
            ok, msg = _grade_one(row["subject_id"], row["notebook_text"])
            st.write(f"{'✅' if ok else '⚠️'} {row['subject_id']}: {msg}")
        prog.progress(1.0, text="Done.")
        st.rerun()

    st.divider()
    st.markdown("#### Re-parse session logs")
    st.caption(
        "Recompute the stored behaviour metrics for every session log — run this "
        "after a parser change or if the Overview shows stale numbers."
    )
    if st.button("Re-parse all logs"):
        logs = list_participant_logs_raw()
        n = 0
        for row in logs:
            m = compute_log_metrics(parse_jsonl(row["raw_jsonl"]))
            ok, _ = save_participant_log(row["subject_id"], row["filename"], row["raw_jsonl"], m)
            n += int(ok)
        st.success(f"Re-parsed {n}/{len(logs)} log(s).")
        st.rerun()

    st.divider()
    with st.expander("Database status"):
        status = supabase_status()
        st.write("Client created:", status["client_created"])
        st.dataframe(
            pd.DataFrame(
                [{"secret": n, **{k: str(v) for k, v in status[n].items()}}
                 for n in ("SUPABASE_URL", "SUPABASE_KEY", "ANTHROPIC_API_KEY")]
            ),
            hide_index=True, width="stretch",
        )
