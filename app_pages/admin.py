"""Admin — the researcher's view of every participant.

Password-gated (set ADMIN_PASSWORD in secrets / .env). Five top-level tabs:
  1. Overview        — one row per participant, filter by condition, CSV export
  2. Participant     — sub-tabs per task/data type: Task plan, Debug (manual vs
                       AI), Ideate (manual vs AI), Notebook grade, Quiz, Verbal
                       assessment, Session timeline, Raw files. Session timeline
                       carries per-turn behaviour tagging (phase / delegation
                       posture / trust) via an "Annotate" button, shown in place
                       as 💬 popovers. Verbal assessment carries a "Fact-check"
                       button instead — it tags each spoken answer's accuracy
                       against the participant's own notebook, not behaviour.
  3. Cohort stats    — outcomes + behaviour overall and split by condition
  4. Notebook report — the visual grading report across all graded notebooks
  5. Grading         — browse/upload any rubric version, grade or re-grade
                       (individually or in bulk) against whichever one you pick;
                       also hosts "Annotate everything pending" (lib/annotate.py),
                       the cross-participant bulk version of the turn tagging above

One page of the multipage app — run via `streamlit run app.py`.
"""

import io

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from build_report import records_from_graded, render_report
from db import (
    annotated_turn_indexes,
    delete_turn_annotations,
    get_participant_bundle,
    get_rubric,
    get_secret,
    list_graded_participant_notebooks,
    list_notebook_section_scores,
    list_participant_logs_raw,
    list_participant_summaries,
    list_participant_transcripts_raw,
    list_pending_grading,
    list_rubrics,
    list_turn_annotations,
    save_participant_file,
    save_participant_log,
    save_participant_notebook,
    save_participant_transcript,
    save_rubric,
    save_turn_annotations,
    set_participant_grading_error,
    supabase_status,
    update_participant_grading,
)
from lib import annotate, cohort, grading, transcript
from lib.llm import get_client
from lib.notebook import notebook_to_text
from lib.quiz_ui import render_quiz_breakdown
from lib.timeline import compute_log_metrics, merge_parsed, parse_jsonl, render_timeline

load_dotenv()

CONDITION_LABEL = cohort.CONDITION_LABEL
ACTIVE_RUBRIC_NAME = get_secret("ACTIVE_RUBRIC_NAME", grading.DEFAULT_RUBRIC_NAME)
DOC_TITLES = {
    "task_plan": "Task plan",
    "debug_manual": "Debug — manual", "debug_ai": "Debug — AI",
    "ideate_manual": "Ideate — manual", "ideate_ai": "Ideate — AI",
}
DOC_SUFFIXES = {
    "task_plan": "task_plan.md",
    "debug_manual": "debug_manual.md", "debug_ai": "debug_ai.md",
    "ideate_manual": "ideate_manual.md", "ideate_ai": "ideate_ai.md",
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


def _grade_one(subject_id: str, notebook_text: str, rubric_name: str) -> tuple[bool, str]:
    """Grade one participant's notebook against the named rubric. Returns (ok, msg)."""
    rubric = get_rubric(rubric_name)
    if rubric is None or not (rubric.get("rubric_csv") or "").strip():
        set_participant_grading_error(subject_id, f"No usable rubric '{rubric_name}'.")
        return False, f"No usable rubric '{rubric_name}'."
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
        subject_id, rubric_name, grading.MODEL, results, total, mx
    )
    return (ok, "graded" if ok else (err or "save failed"))


def _annotate_log_for(
    subject_id: str, raw_jsonl: str, log_filename: str, force: bool = False
) -> tuple[bool, str]:
    """Tag every not-yet-annotated turn in one of a participant's session logs —
    or, with force=True, every turn, overwriting whatever's already there."""
    existing = (
        frozenset() if force
        else frozenset(annotated_turn_indexes(subject_id, "log", log_filename))
    )
    try:
        new_annos = annotate.annotate_log(get_client(), parse_jsonl(raw_jsonl), existing)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    if not new_annos:
        return True, "nothing new to annotate"
    ok, err = save_turn_annotations(
        subject_id, "log", new_annos, annotate.MODEL, log_filename=log_filename
    )
    verb = "re-annotated" if force else "annotated"
    return (ok, f"{verb} {len(new_annos)} turn(s)" if ok else (err or "save failed"))


def _annotate_transcript_for(
    subject_id: str, pairs: list[dict], notebook_text: str | None, force: bool = False
) -> tuple[bool, str]:
    """Fact-check every not-yet-annotated Q/A pair in one participant's verbal
    transcript against their own notebook — or, with force=True, every pair,
    overwriting whatever's already there."""
    if not notebook_text:
        return False, "no notebook on file to check the answers against"
    existing = frozenset() if force else frozenset(annotated_turn_indexes(subject_id, "transcript"))
    try:
        new_annos = annotate.annotate_transcript(get_client(), pairs, notebook_text, existing)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    if not new_annos:
        return True, "nothing new to annotate"
    ok, err = save_turn_annotations(subject_id, "transcript", new_annos, annotate.MODEL)
    verb = "re-annotated" if force else "annotated"
    return (ok, f"{verb} {len(new_annos)} turn(s)" if ok else (err or "save failed"))


def _reannotate_one_log_turn(
    subject_id: str, raw_jsonl: str, log_filename: str, turn: int
) -> tuple[bool, str]:
    """Re-annotate exactly one turn in one of a participant's session logs,
    leaving every other turn's existing tag untouched."""
    parsed = parse_jsonl(raw_jsonl)
    all_turns = {e["turn"] for e in parsed["events"] if e["lane"] == "User Prompt"}
    skip = frozenset(all_turns - {turn})
    try:
        new_annos = annotate.annotate_log(get_client(), parsed, skip)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    if not new_annos:
        return False, "turn not found"
    ok, err = save_turn_annotations(
        subject_id, "log", new_annos, annotate.MODEL, log_filename=log_filename
    )
    return (ok, "re-annotated" if ok else (err or "save failed"))


def _reannotate_one_transcript_pair(
    subject_id: str, pairs: list[dict], notebook_text: str, idx: int
) -> tuple[bool, str]:
    """Re-check exactly one Q/A pair against the notebook, leaving every other
    pair's existing tag untouched."""
    skip = frozenset(set(range(len(pairs))) - {idx})
    try:
        new_annos = annotate.annotate_transcript(get_client(), pairs, notebook_text, skip)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    if not new_annos:
        return False, "pair not found"
    ok, err = save_turn_annotations(subject_id, "transcript", new_annos, annotate.MODEL)
    return (ok, "re-checked" if ok else (err or "save failed"))


def _decode_upload(up) -> str:
    try:
        return up.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError:
        return up.getvalue().decode("latin-1")


def _doc_upload_widget(sid: str, doc_type: str, label: str) -> None:
    """A small re-upload control for one of the five reflection docs — always
    saved under the canonical filename regardless of what the local file is
    called, so it lines up with what the rest of Admin expects to see."""
    up = st.file_uploader(f"Replace {label}", type=["md"], key=f"up_{doc_type}")
    if up is not None and st.button(f"Save {label}", key=f"save_{doc_type}"):
        content = _decode_upload(up)
        filename = f"{sid}_{DOC_SUFFIXES[doc_type]}"
        ok, err = save_participant_file(sid, doc_type, filename, content)
        (st.success if ok else st.error)("Saved." if ok else f"Not saved: {err}")
        if ok:
            st.rerun()


def _notebook_upload_widget(sid: str) -> None:
    """Replace the participant's notebook. Clears existing verbal-assessment
    fact-checks (they were checked against the old notebook content) but leaves
    any existing grade alone -- it's just stale until Re-grade is clicked."""
    up = st.file_uploader("Replace notebook (.ipynb)", type=["ipynb"], key="up_notebook")
    if up is not None and st.button("Save notebook", key="save_notebook"):
        try:
            notebook_text = notebook_to_text(up.getvalue())
        except Exception as e:
            st.error(f"Could not read notebook: {e}")
            return
        if not notebook_text.strip():
            st.error("Notebook appears to be empty.")
            return
        ok, err = save_participant_notebook(sid, f"{sid}_notebook.ipynb", notebook_text)
        if ok:
            delete_turn_annotations(sid, "transcript")
            st.success(
                "Saved. Any existing grade is now stale (use Re-grade below) and "
                "verbal-assessment fact-checks (if any) were cleared, since they "
                "were checked against the old notebook."
            )
            st.rerun()
        else:
            st.error(f"Not saved: {err}")


tab_overview, tab_detail, tab_cohort, tab_report, tab_grading = st.tabs(
    ["1 · Overview", "2 · Participant", "3 · Cohort stats",
     "4 · Notebook report", "5 · Grading & rubric"]
)

_section_rows = list_notebook_section_scores()  # (participant, section) grades; used in tabs 2 & 3
_all_rubric_names = list_rubrics()  # every saved rubric version; used in tabs 2 & 5
_rubric_options = _all_rubric_names or [ACTIVE_RUBRIC_NAME]


def _rubric_index(preferred: str | None) -> int:
    """Index of `preferred` in _rubric_options, falling back to ACTIVE_RUBRIC_NAME, then 0."""
    for candidate in (preferred, ACTIVE_RUBRIC_NAME):
        if candidate in _rubric_options:
            return _rubric_options.index(candidate)
    return 0

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
            "n_tool_calls", "n_edits", "n_logs", "has_transcript",
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
        nb = bundle["notebook"]
        logs = bundle["logs"]

        (
            sub_taskplan, sub_debug, sub_ideate,
            sub_notebook, sub_quiz, sub_verbal, sub_timeline, sub_files,
        ) = st.tabs([
            "Task plan", "Debug: manual vs AI", "Ideate: manual vs AI",
            "Notebook grade", "Quiz", "Verbal assessment", "Session timeline", "Raw files",
        ])

        with sub_taskplan:
            if "task_plan" in docs:
                st.markdown(docs["task_plan"]["content"])
            else:
                st.caption("No task plan on file.")
            with st.expander("Upload / replace"):
                _doc_upload_widget(sid, "task_plan", DOC_TITLES["task_plan"])

        for sub, left, right in (
            (sub_debug, "debug_manual", "debug_ai"),
            (sub_ideate, "ideate_manual", "ideate_ai"),
        ):
            with sub:
                c1, c2 = st.columns(2)
                for col, key in ((c1, left), (c2, right)):
                    with col:
                        st.markdown(f"**{DOC_TITLES[key]}**")
                        st.markdown(docs[key]["content"] if key in docs else "_missing_")
                        with st.expander("Upload / replace"):
                            _doc_upload_widget(sid, key, DOC_TITLES[key])

        with sub_notebook:
            if nb and nb.get("results"):
                st.metric(
                    "Total",
                    f"{nb.get('total_score')} / {nb.get('max_score')}"
                    + (f"  ({100 * nb['total_score'] / nb['max_score']:.0f}%)"
                       if nb.get("max_score") else ""),
                )
                st.caption(f"Graded against rubric `{nb.get('rubric_name', '?')}`.")
                cohort.render_participant_notebook_sections(nb["results"], _section_rows, sid)
                with st.expander("Per-criterion detail"):
                    gdf = pd.DataFrame(nb["results"])[
                        ["section", "criterion", "score", "max_pts", "reasoning"]
                    ]
                    st.dataframe(gdf, hide_index=True, width="stretch")
                if nb.get("notebook_text"):
                    with st.expander("Notebook transcript (as graded)"):
                        st.code(nb["notebook_text"])
                c_pick, c_go = st.columns([3, 1])
                pick = c_pick.selectbox(
                    "Re-grade against", _rubric_options,
                    index=_rubric_index(nb.get("rubric_name")), key="regrade_rubric_pick",
                )
                if c_go.button("Re-grade", key="regrade_detail"):
                    with st.spinner(f"Grading against `{pick}`…"):
                        ok, msg = _grade_one(sid, nb["notebook_text"], pick)
                    (st.success if ok else st.error)(msg)
                    st.rerun()
            elif nb:
                st.caption("Notebook stored but not graded yet.")
                pick = st.selectbox(
                    "Grade against", _rubric_options,
                    index=_rubric_index(None), key="grade_now_rubric_pick",
                )
                if st.button("Grade now", key="grade_now_detail"):
                    with st.spinner(f"Grading against `{pick}`…"):
                        ok, msg = _grade_one(sid, nb["notebook_text"], pick)
                    (st.success if ok else st.error)(msg)
                    st.rerun()
                if nb.get("notebook_text"):
                    with st.expander("Notebook transcript (ungraded)"):
                        st.code(nb["notebook_text"])
            else:
                st.caption("No notebook on file.")
            with st.expander("Upload / replace notebook"):
                _notebook_upload_widget(sid)

        with sub_quiz:
            if bundle["quiz"]:
                render_quiz_breakdown(bundle["quiz"])
            else:
                st.caption("No quiz result on file.")

        with sub_verbal:
            tr = bundle["transcript"]
            if tr:
                pairs = tr.get("parsed") or []
                notebook_text = nb.get("notebook_text") if nb else None
                anno_map = {
                    a["turn_index"]: a for a in list_turn_annotations(sid) if a["source"] == "transcript"
                }
                if pairs:
                    n_missing = len(pairs) - len(anno_map)
                    if not notebook_text:
                        st.caption("No notebook on file — can't fact-check answers against it yet.")
                    c_btn, c_force = st.columns([2, 2])
                    force_tr = c_force.checkbox("Re-check everything", key="force_reannotate_tr")
                    n_target = len(pairs) if force_tr else n_missing
                    if c_btn.button(
                        f"🏷️ {'Re-check' if force_tr else 'Fact-check'} ({n_target})"
                        if n_target else "🏷️ Fact-check (up to date)",
                        key="annotate_tr", disabled=n_target == 0 or not notebook_text,
                    ):
                        with st.spinner("Checking answers against the notebook…"):
                            ok, msg = _annotate_transcript_for(sid, pairs, notebook_text, force=force_tr)
                        (st.success if ok else st.error)(msg)
                        st.rerun()
                _ACCURACY_ICON = {
                    "accurate": "✅", "partially_accurate": "⚠️",
                    "inaccurate": "❌", "unverifiable": "❔",
                }
                for i, pr in enumerate(pairs):
                    st.markdown(f"**`{pr.get('timestamp', '')}` · {pr.get('question', '')}**")
                    st.markdown(pr.get("answer") or "_(no answer)_")
                    a = anno_map.get(i)
                    if a:
                        icon = _ACCURACY_ICON.get(a["accuracy"], "🏷️")
                        label = a["accuracy"].replace("_", " ").title()
                        st.caption(f"{icon} **{label}** — {a.get('reasoning') or ''}")
                        if st.button("🔄 Re-check this answer", key=f"reanno_tr_{i}"):
                            with st.spinner("Re-checking…"):
                                ok, msg = _reannotate_one_transcript_pair(
                                    sid, pairs, notebook_text, i
                                )
                            (st.success if ok else st.error)(msg)
                            st.rerun()
                    st.divider()
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

        with sub_timeline:
            if not logs:
                st.caption("No session log on file.")
            else:
                if len(logs) > 1:
                    merged = merge_parsed([parse_jsonl(lg["raw_jsonl"]) for lg in logs])
                    cm = compute_log_metrics(merged)
                    with st.expander(f"Combined across {len(logs)} logs", expanded=False):
                        mc0 = st.columns(4)
                        dur0 = (cm.get("session_duration_s") or 0) / 60
                        mc0[0].metric("Duration", f"{dur0:.1f} min")
                        if cm.get("format") == "history":
                            mc0[1].metric("Prompts", cm.get("n_substantive_prompts", 0))
                            mc0[2].metric("Sessions", cm.get("n_sessions", 0))
                            g0 = cm.get("median_inter_prompt_gap_s")
                            mc0[3].metric("Median gap", f"{g0:.0f}s" if g0 is not None else "—")
                        else:
                            mc0[1].metric("Tool calls", cm.get("n_tool_calls", 0))
                            mc0[2].metric("File edits", cm.get("n_edits", 0))
                            ttf0 = cm.get("time_to_first_tool_call_s")
                            mc0[3].metric("To 1st tool", f"{ttf0:.0f}s" if ttf0 is not None else "—")

                log_names = [lg["filename"] for lg in logs]
                sel = st.selectbox(
                    "Log file", range(len(logs)), index=len(logs) - 1,  # most recent
                    format_func=lambda i: log_names[i], key="log_pick",
                )
                log = logs[sel]
                log_filename = log["filename"]

                parsed_log = parse_jsonl(log["raw_jsonl"])
                metrics = compute_log_metrics(parsed_log)
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
                render_timeline(log["raw_jsonl"], key=f"{sid}_{log_filename}")

                st.markdown("#### Turn annotations")
                prompt_events = [e for e in parsed_log["events"] if e["lane"] == "User Prompt"]
                anno_map = {
                    a["turn_index"]: a for a in list_turn_annotations(sid)
                    if a["source"] == "log" and a.get("log_filename") == log_filename
                }
                n_missing = len(prompt_events) - len(anno_map)
                c_btn, c_force = st.columns([2, 2])
                force_log = c_force.checkbox(
                    "Re-annotate everything", key=f"force_reannotate_log_{log_filename}"
                )
                n_target = len(prompt_events) if force_log else n_missing
                if c_btn.button(
                    f"🏷️ {'Re-annotate' if force_log else 'Annotate'} ({n_target})"
                    if n_target else "🏷️ Annotate (up to date)",
                    key=f"annotate_log_{log_filename}", disabled=n_target == 0,
                ):
                    with st.spinner("Annotating…"):
                        ok, msg = _annotate_log_for(
                            sid, log["raw_jsonl"], log_filename, force=force_log
                        )
                    (st.success if ok else st.error)(msg)
                    st.rerun()
                if anno_map:
                    for e in prompt_events:
                        a = anno_map.get(e["turn"])
                        if not a:
                            continue
                        c_lbl, c_tag = st.columns([6, 1])
                        c_lbl.caption(
                            f"Turn {e['turn']} · **{a['phase']}** · "
                            f"{a['delegation_posture']} · {a['trust_behavior']}"
                        )
                        with c_tag.popover("💬"):
                            st.write(a.get("turn_text") or "")
                            st.caption(a.get("reasoning") or "")
                            if st.button(
                                "🔄 Re-annotate this turn",
                                key=f"reanno_log_{log_filename}_{e['turn']}",
                            ):
                                with st.spinner("Re-annotating…"):
                                    ok, msg = _reannotate_one_log_turn(
                                        sid, log["raw_jsonl"], log_filename, e["turn"]
                                    )
                                (st.success if ok else st.error)(msg)
                                st.rerun()
                else:
                    st.caption("No annotations yet — click Annotate above.")

            st.divider()
            st.markdown("#### Add a log")
            st.caption(
                "Adds another session-log file for this participant (e.g. a separate "
                "work session) — the uploaded file's own name is used as-is. "
                "Re-uploading a name already on file replaces just that log."
            )
            up = st.file_uploader("Session log (.jsonl)", type=["jsonl"], key="up_log")
            if up is not None and st.button("Save log", key="save_log"):
                raw_jsonl = up.getvalue().decode("utf-8", errors="replace")
                new_metrics = compute_log_metrics(parse_jsonl(raw_jsonl))
                ok, err = save_participant_log(sid, up.name, raw_jsonl, new_metrics)
                (st.success if ok else st.error)("Saved." if ok else f"Not saved: {err}")
                if ok:
                    st.rerun()

        with sub_files:
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
            for lg in logs:
                st.download_button(
                    f"⬇️ {lg['filename']}", data=lg["raw_jsonl"],
                    file_name=lg["filename"], mime="application/json", key=f"dl_log_{lg['filename']}",
                )
            if not bundle["files"] and not nb and not logs:
                st.caption("No files on record.")

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
        f"New submissions are auto-graded against the rubric named **`{ACTIVE_RUBRIC_NAME}`** "
        "(override with the `ACTIVE_RUBRIC_NAME` secret). Every saved rubric version stays "
        "on file below — browse any of them, and (re-)grade against whichever one you pick."
    )
    # The save button below calls st.rerun() right after saving — without this,
    # the success/error message it shows gets wiped before it ever renders, so
    # a save silently *looks* like it did nothing even when it worked.
    _flash = st.session_state.pop("_rubric_save_msg", None)
    if _flash is not None:
        (st.success if _flash[0] else st.error)(_flash[1])

    active = get_rubric(ACTIVE_RUBRIC_NAME)
    if active and (active.get("rubric_csv") or "").strip():
        _upd = active.get("updated_at")
        st.success(
            f"Rubric `{ACTIVE_RUBRIC_NAME}` is configured"
            + (f" · last saved {_upd}" if _upd else "") + "."
        )
    else:
        st.error(
            f"No usable rubric `{ACTIVE_RUBRIC_NAME}` — new submissions will save but "
            "stay ungraded until you upload one below and re-grade them."
        )

    st.markdown("#### Browse saved rubrics")
    if not _all_rubric_names:
        st.caption("No rubrics saved yet.")
    else:
        view_pick = st.selectbox(
            "Rubric version", _rubric_options, index=_rubric_index(None), key="view_rubric_pick",
        )
        viewed = active if view_pick == ACTIVE_RUBRIC_NAME else get_rubric(view_pick)
        if viewed and (viewed.get("rubric_csv") or "").strip():
            _vupd = viewed.get("updated_at")
            st.caption(
                f"`{view_pick}`" + (" · **active**" if view_pick == ACTIVE_RUBRIC_NAME else "")
                + (f" · last saved {_vupd}" if _vupd else "")
            )
            with st.expander("Task description"):
                st.markdown(viewed.get("task") or "_(empty)_")
            try:
                st.dataframe(
                    pd.read_csv(io.StringIO(viewed["rubric_csv"])), hide_index=True, width="stretch"
                )
            except Exception:
                st.code(viewed["rubric_csv"], language="csv")
        else:
            st.caption(f"`{view_pick}` has no usable CSV.")

    st.markdown("#### Upload / replace a rubric")
    if _all_rubric_names:
        st.caption("Saved rubrics: " + ", ".join(f"`{r}`" for r in _all_rubric_names))
    name = st.text_input(
        "Rubric name — an existing name replaces that version; a new name adds one",
        value=ACTIVE_RUBRIC_NAME,
    )
    name = name.strip() or ACTIVE_RUBRIC_NAME
    # Look up whatever's already stored under this name so "replace" doesn't
    # silently blank a customised task back to the generic template. Keyed on
    # `name` so the box actually refreshes when you switch which rubric you're
    # editing, instead of keeping whatever was typed for a different one.
    target = active if name == ACTIVE_RUBRIC_NAME else get_rubric(name)
    task_default = (target or {}).get("task") or grading.DEFAULT_TASK
    task_text = st.text_area(
        "Task description (graded against)", value=task_default, height=220, key=f"rubric_task_{name}"
    )
    up = st.file_uploader("Rubric CSV", type=["csv"])
    if up is not None and st.button("Save rubric", type="primary"):
        try:
            csv_text = up.getvalue().decode("utf-8-sig")
        except UnicodeDecodeError:
            csv_text = up.getvalue().decode("latin-1")
        ok, err = save_rubric(name, task_text, csv_text)
        st.session_state["_rubric_save_msg"] = (
            ok, f"Saved `{name}` ({len(csv_text.splitlines())} CSV line(s))." if ok
            else f"Not saved: {err}"
        )
        st.rerun()

    st.divider()
    st.markdown("#### Grade participants")
    grade_pick = st.selectbox(
        "Grade against", _rubric_options, index=_rubric_index(None), key="bulk_grade_rubric_pick",
    )
    regrade_all = st.checkbox(
        f"Include already-graded notebooks (re-grade everyone against `{grade_pick}`)"
    )
    pending = list_pending_grading(include_graded=regrade_all)
    st.caption(
        f"{len(pending)} participant notebook(s) "
        + ("on file." if regrade_all else "pending or errored.")
    )
    if pending and st.button(f"Grade {len(pending)} notebook(s) against `{grade_pick}`", type="primary"):
        prog = st.progress(0.0, text="Starting…")
        for i, row in enumerate(pending):
            prog.progress(i / len(pending), text=f"Grading {row['subject_id']}…")
            ok, msg = _grade_one(row["subject_id"], row["notebook_text"], grade_pick)
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
    st.markdown("#### Annotate everything pending")
    st.caption(
        "Tags every not-yet-annotated turn — a session-log prompt or a verbal-"
        "assessment Q/A pair — across every participant. One LLM call per turn; "
        "already-annotated turns are skipped, so this is safe to re-run."
    )
    force_all = st.checkbox(
        "Re-annotate everything (overwrite already-tagged turns too)", key="force_reannotate_all"
    )
    _pending_logs = []
    for row in list_participant_logs_raw():
        n_prompts = sum(
            1 for e in parse_jsonl(row["raw_jsonl"])["events"] if e["lane"] == "User Prompt"
        )
        n_target = n_prompts if force_all else (
            n_prompts - len(annotated_turn_indexes(row["subject_id"], "log", row["filename"]))
        )
        if n_target:
            _pending_logs.append((row, n_target))
    _pending_trs = []
    _skipped_trs_no_notebook = 0
    for row in list_participant_transcripts_raw():
        pairs = row.get("parsed") or []
        if not pairs:
            continue
        if not row.get("notebook_text"):
            _skipped_trs_no_notebook += 1
            continue
        n_target = len(pairs) if force_all else (
            len(pairs) - len(annotated_turn_indexes(row["subject_id"], "transcript"))
        )
        if n_target:
            _pending_trs.append((row, n_target))
    _total_calls = sum(n for _, n in _pending_logs) + sum(n for _, n in _pending_trs)
    st.caption(
        f"{len(_pending_logs)} log(s) and {len(_pending_trs)} transcript(s) "
        + ("to re-annotate" if force_all else "have new turns")
        + f" — about {_total_calls} call(s) total."
        + (f" ({_skipped_trs_no_notebook} transcript(s) skipped — no notebook to "
           "fact-check against.)" if _skipped_trs_no_notebook else "")
    )
    _bulk_label = "Re-annotate" if force_all else "Annotate"
    if _total_calls and st.button(
        f"🏷️ {_bulk_label} everything pending ({_total_calls} call(s))", type="primary"
    ):
        prog = st.progress(0.0, text="Starting…")
        n_steps = len(_pending_logs) + len(_pending_trs)
        step = 0
        for row, _n in _pending_logs:
            step += 1
            prog.progress(step / n_steps, text=f"Annotating {row['subject_id']}'s log…")
            ok, msg = _annotate_log_for(
                row["subject_id"], row["raw_jsonl"], row["filename"], force=force_all
            )
            st.write(f"{'✅' if ok else '⚠️'} {row['subject_id']} (log): {msg}")
        for row, _n in _pending_trs:
            step += 1
            prog.progress(step / n_steps, text=f"Fact-checking {row['subject_id']}'s transcript…")
            ok, msg = _annotate_transcript_for(
                row["subject_id"], row.get("parsed") or [], row.get("notebook_text"), force=force_all
            )
            st.write(f"{'✅' if ok else '⚠️'} {row['subject_id']} (transcript): {msg}")
        prog.progress(1.0, text="Done.")
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
