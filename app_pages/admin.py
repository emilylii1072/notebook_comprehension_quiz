"""Admin — the researcher's view of every participant.

Password-gated (set ADMIN_PASSWORD in secrets / .env). Seven top-level tabs:
  1. Overview        — one row per participant, filter by condition, CSV export
  2. Participant     — sub-tabs per task/data type: Surveys, Task timing, Task
                       plan, Ideate, Debug, Notebook grade, Quiz, Verbal
                       assessment, Session timeline, Raw files.
                       Debug and Ideate each have an Autograde block. Debug: the
                       write-up is read against the buggy notebook the admin uploads
                       under Task instructions; each bug the participant reported
                       scores +1 if it is a real bug and +1 if the fix is valid.
                       Ideate: participants upload nothing; the
                       admin uploads a transcript of their spoken pitch, scored
                       against an admin-uploaded rubric (Task instructions).
                       Task timing shows how long each timed task actually took
                       against its allowance, plus any extra files attached to
                       the main task. Ideate/Debug are one document each; a
                       participant run under the earlier manual-vs-AI protocol
                       still shows those split documents underneath.
                       Surveys shows this one participant's pre-/post-survey
                       (lib/surveys.py, transcribed from the study's Qualtrics
                       PDFs) with per-item timing and ✅/❌ on the scored
                       knowledge items, read-only. Session timeline
                       carries per-turn behaviour tagging (phase / delegation
                       posture / trust) via an "Annotate" button, shown in place
                       as 💬 popovers. Verbal assessment carries a "Fact-check"
                       button instead — it grades each spoken answer 0-2 (checked
                       against the participant's own notebook where it relies on
                       it; "inapplicable" if the notebook is empty), not behaviour.
  3. Cohort stats    — outcomes + behaviour overall and split by condition
  4. Survey results  — the pre/post surveys across every participant
                       (lib/survey_stats.py): coverage and timing, knowledge
                       gain against lib.surveys.ANSWER_KEY, attitude shift,
                       workload, self-assessed vs rubric grade, background,
                       free text, and a long-format CSV export
  5. Task instructions — the markdown each task screen shows the participant:
                       a shared main-task brief plus one condition-instructions
                       document per condition (shown together on one page) and
                       the shared ideation and debugging ones (lib/tasks.py).
                       Participants cannot start a task whose document is
                       missing.
  6. Notebook report — the visual grading report across all graded notebooks
  7. Grading         — browse/upload any rubric version, grade or re-grade
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
    delete_participant_file,
    delete_participant_extra_file,
    delete_participant_log,
    delete_participant_notebook,
    delete_task_instruction,
    delete_turn_annotations,
    extra_file_bytes,
    get_participant_bundle,
    get_rubric,
    get_secret,
    get_task_grade,
    get_task_instruction,
    list_graded_participant_notebooks,
    list_notebook_section_scores,
    list_participant_logs_raw,
    list_participant_summaries,
    list_participant_survey_rows,
    list_participant_transcripts_raw,
    list_pending_grading,
    list_rubrics,
    list_task_instructions,
    list_turn_annotations,
    save_participant_file,
    save_participant_log,
    save_participant_notebook,
    save_participant_transcript,
    save_rubric,
    save_task_grade,
    save_task_instruction,
    save_turn_annotations,
    set_participant_grading_error,
    supabase_status,
    update_participant_grading,
)
from lib import (
    annotate, cohort, debug_grading, grading, ideate_grading, survey_stats, tasks, transcript,
)
from lib.llm import get_client
from lib.notebook import notebook_has_content, notebook_to_html, notebook_to_text
from lib.quiz_ui import render_quiz_breakdown
from lib.survey_ui import render_survey_breakdown
from lib.timeline import (
    compute_log_metrics,
    merge_parsed,
    parse_jsonl,
    render_parsed_timeline,
)

load_dotenv()

CONDITION_LABEL = cohort.CONDITION_LABEL
ACTIVE_RUBRIC_NAME = get_secret("ACTIVE_RUBRIC_NAME", grading.DEFAULT_RUBRIC_NAME)
DOC_TITLES = {
    "task_plan": "Task plan",
    # A transcript of the participant's spoken pitch, uploaded by the admin. Runs
    # before the pitch was transcribed left a written write-up here instead.
    "ideate": "Idea pitch transcript",
    "debug": "Debugging write-up",
    # Collected under the earlier manual-vs-AI protocol. Nothing writes these
    # any more, but participants run back then still have them, so they stay
    # displayable (and replaceable) here.
    "debug_manual": "Debug — manual (legacy)", "debug_ai": "Debug — AI (legacy)",
    "ideate_manual": "Ideate — manual (legacy)", "ideate_ai": "Ideate — AI (legacy)",
}
DOC_SUFFIXES = {
    "task_plan": "task_plan.md",
    "ideate": "ideate.md",
    "debug": "debug.md",
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


def _timing_seconds(timing: dict) -> float | None:
    """How long a task took: finished_at - started_at, or None while it's still
    open. Supabase hands these back as ISO-8601 strings."""
    from datetime import datetime

    def _parse(raw):
        if not raw:
            return None
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            return None

    start, end = _parse(timing.get("started_at")), _parse(timing.get("finished_at"))
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def _grade_one(subject_id: str, notebook_text: str, rubric_name: str) -> tuple[bool, str]:
    """Grade one participant's notebook against the named rubric. Returns (ok, msg)."""
    rubric = get_rubric(rubric_name)
    if rubric is None or not (rubric.get("rubric_csv") or "").strip():
        set_participant_grading_error(subject_id, f"No usable rubric '{rubric_name}'.")
        return False, f"No usable rubric '{rubric_name}'."
    try:
        results = grading.grade_notebook(
            get_client(), rubric["task"], rubric["rubric_csv"], notebook_text,
            instructions=rubric.get("grading_instructions"),
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


def _annotate_log_core(
    subject_id: str, raw_jsonl: str, log_filename: str, force: bool
) -> tuple[bool, int, str | None]:
    """Tag every not-yet-annotated turn in one session log (or every turn, with
    force=True). Returns (ok, n_annotated, error)."""
    existing = (
        frozenset() if force
        else frozenset(annotated_turn_indexes(subject_id, "log", log_filename))
    )
    try:
        new_annos = annotate.annotate_log(get_client(), parse_jsonl(raw_jsonl), existing)
    except Exception as e:
        return False, 0, f"{type(e).__name__}: {e}"
    if not new_annos:
        return True, 0, None
    ok, err = save_turn_annotations(
        subject_id, "log", new_annos, annotate.MODEL, log_filename=log_filename
    )
    return ok, (len(new_annos) if ok else 0), (None if ok else (err or "save failed"))


def _annotate_log_for(
    subject_id: str, raw_jsonl: str, log_filename: str, force: bool = False
) -> tuple[bool, str]:
    """Tag every not-yet-annotated turn in one of a participant's session logs —
    or, with force=True, every turn, overwriting whatever's already there."""
    ok, n, err = _annotate_log_core(subject_id, raw_jsonl, log_filename, force)
    if not ok:
        return False, err
    verb = "re-annotated" if force else "annotated"
    return True, f"{verb} {n} turn(s)" if n else "nothing new to annotate"


def _annotate_all_logs_for(
    subject_id: str, logs: list[dict], force: bool = False
) -> tuple[bool, str]:
    """Tag every not-yet-annotated turn across ALL of a participant's session
    logs (or every turn, with force=True), combined into a single result."""
    total = 0
    errors = []
    for lg in logs:
        ok, n, err = _annotate_log_core(subject_id, lg["raw_jsonl"], lg["filename"], force)
        if not ok:
            errors.append(f"{lg['filename']}: {err}")
        else:
            total += n
    if errors:
        return False, "; ".join(errors)
    verb = "re-annotated" if force else "annotated"
    return True, (f"{verb} {total} turn(s) across {len(logs)} log(s)" if total else "nothing new to annotate")


def _has_text(row: dict | None) -> bool:
    return bool(row and (row.get("content") or "").strip())


def _grade_task_for(subject_id: str, task_key: str, text: str) -> tuple[bool, str]:
    """Autograde one participant's debugging write-up (tasks.DEBUG) or ideation pitch
    transcript (tasks.IDEATE) against the admin's grading material for that task
    (Task instructions tab). Returns (ok, msg)."""
    try:
        if task_key == tasks.DEBUG:
            nb_row = get_task_instruction(tasks.DEBUG_NOTEBOOK_KEY)
            if not nb_row:
                return False, "upload the buggy notebook under Task instructions first"
            graded = debug_grading.grade_debug_writeup(get_client(), nb_row["content"], text)
            results, notes, model = graded["results"], graded["notes"], debug_grading.MODEL
        else:
            rubric_row = get_task_instruction(tasks.IDEATE_RUBRIC_KEY)
            if not _has_text(rubric_row):
                return False, "upload the ideation rubric under Task instructions first"
            task_row = get_task_instruction(tasks.IDEATE)
            results = ideate_grading.grade_pitch(
                get_client(), (task_row or {}).get("content") or "", rubric_row["content"], text
            )
            notes, model = "", ideate_grading.MODEL
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    total = round(sum(r["score"] for r in results), 2)
    mx = round(sum(r["max_pts"] for r in results), 2)
    ok, err = save_task_grade(subject_id, task_key, results, total, mx, notes, model)
    if not ok and err and "participant_task_grades" in err:
        err += " — run the participant_task_grades block at the end of supabase_schema.sql in Supabase"
    return (ok, f"graded: {total:g} / {mx:g}" if ok else (err or "save failed"))


def _render_graded_rubric(results: list[dict], sid: str, name: str) -> None:
    """A filled-in rubric as readable rows — score, criterion, and the grader's full
    reasoning, grouped by section — rather than a table that clips long cells, plus
    a CSV download of the same rows."""
    by_section: dict[str, list[dict]] = {}
    for r in results:
        by_section.setdefault((r.get("section") or "").strip(), []).append(r)
    for section, items in by_section.items():
        if section:
            got = sum(float(i.get("score") or 0) for i in items)
            mx = sum(float(i.get("max_pts") or 0) for i in items)
            st.markdown(f"**{section}** — {got:g} / {mx:g}")
        for i in items:
            c_score, c_text = st.columns([1, 7])
            c_score.markdown(f"**{float(i.get('score') or 0):g} / {float(i.get('max_pts') or 0):g}**")
            c_text.markdown(f"**{i.get('criterion') or ''}**")
            if i.get("reasoning"):
                c_text.markdown(i["reasoning"])
    csv = pd.DataFrame(results)[["section", "criterion", "score", "max_pts", "reasoning"]]
    st.download_button(
        "⬇️ Download graded rubric (CSV)", data=csv.to_csv(index=False).encode("utf-8"),
        file_name=f"{sid}_{name}_graded.csv", mime="text/csv", key=f"dl_graded_{name}_{sid}",
    )


def _render_task_grade(sid: str, task_key: str, doc: dict | None) -> None:
    """The Autograde block under a participant's debugging write-up or ideation pitch
    transcript: the stored grade if any, and a (re-)grade button once the admin's
    grading material for that task is in place."""
    if task_key == tasks.DEBUG:
        ready = get_task_instruction(tasks.DEBUG_NOTEBOOK_KEY) is not None
        setup = (
            "To autograde, upload the buggy notebook under Task instructions → "
            "Debugging task — grading material."
        )
        missing_doc = "No debugging write-up to grade."
    else:
        ready = _has_text(get_task_instruction(tasks.IDEATE_RUBRIC_KEY))
        setup = (
            "To autograde, upload the ideation rubric under Task instructions → "
            "Idea generation task — grading rubric."
        )
        missing_doc = "No pitch transcript to grade — upload one above."
    st.divider()
    st.markdown("#### Autograde")
    grade = get_task_grade(sid, task_key)
    if grade:
        st.metric("Total", f"{grade['total_score']:g} / {grade['max_score']:g}")
        if grade["results"]:
            _render_graded_rubric(grade["results"], sid, task_key)
        else:
            st.caption("No bugs were reported in this write-up.")
        if grade.get("notes"):
            st.caption(f"Other reports: {grade['notes']}")
        st.caption(f"Graded {grade.get('graded_at') or ''} against the notebook as it stood then.")
    if not ready:
        st.caption(setup)
    elif doc is None or not (doc.get("content") or "").strip():
        st.caption(missing_doc)
    elif st.button("Re-grade" if grade else "Grade", key=f"grade_{task_key}"):
        with st.spinner("Grading…"):
            ok, msg = _grade_task_for(sid, task_key, doc["content"])
        (st.success if ok else st.error)(msg)
        if ok:
            st.rerun()


def _annotate_transcript_for(
    subject_id: str, pairs: list[dict], notebook_text: str | None, force: bool = False
) -> tuple[bool, str]:
    """Grade every not-yet-annotated Q/A pair in one participant's verbal
    transcript (0-2, against their own notebook where an answer relies on it) — or,
    with force=True, every pair, overwriting whatever's already there. An empty or
    missing notebook is fine: notebook-dependent answers just come back inapplicable."""
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


def _doc_upload_widget(
    sid: str, doc_type: str, label: str, exists: bool, types: tuple[str, ...] = ("md",)
) -> None:
    """A small re-upload / delete control for one of the five reflection docs —
    uploads always save under the canonical filename regardless of what the
    local file is called, so it lines up with what the rest of Admin expects."""
    up = st.file_uploader(f"Replace {label}", type=list(types), key=f"up_{doc_type}")
    if up is not None and st.button(f"Save {label}", key=f"save_{doc_type}"):
        content = _decode_upload(up)
        filename = f"{sid}_{DOC_SUFFIXES[doc_type]}"
        ok, err = save_participant_file(sid, doc_type, filename, content)
        (st.success if ok else st.error)("Saved." if ok else f"Not saved: {err}")
        if ok:
            st.rerun()
    if exists:
        c_conf, c_del = st.columns([2, 1])
        confirm = c_conf.checkbox("Confirm delete", key=f"confirm_del_{doc_type}")
        if c_del.button(f"🗑️ Delete {label}", key=f"del_{doc_type}", disabled=not confirm):
            ok, err = delete_participant_file(sid, doc_type)
            (st.success if ok else st.error)("Deleted." if ok else f"Not deleted: {err}")
            if ok:
                st.rerun()


def _notebook_upload_widget(sid: str, exists: bool) -> None:
    """Replace or delete the participant's notebook. Either way, clears existing
    verbal-assessment fact-checks (they were checked against the old notebook
    content); replacing leaves any existing grade alone -- it's just stale until
    Re-grade is clicked."""
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
    if exists:
        c_conf, c_del = st.columns([2, 1])
        confirm = c_conf.checkbox("Confirm delete", key="confirm_del_notebook")
        if c_del.button("🗑️ Delete notebook", key="del_notebook", disabled=not confirm):
            ok, err = delete_participant_notebook(sid)
            (st.success if ok else st.error)("Deleted." if ok else f"Not deleted: {err}")
            if ok:
                st.rerun()


(
    tab_overview, tab_detail, tab_cohort, tab_surveys,
    tab_tasks, tab_report, tab_grading,
) = st.tabs(
    ["1 · Overview", "2 · Participant", "3 · Cohort stats", "4 · Survey results",
     "5 · Task instructions", "6 · Notebook report", "7 · Grading & rubric"]
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
            sub_surveys, sub_timings, sub_taskplan, sub_ideate, sub_debug,
            sub_notebook, sub_quiz, sub_verbal, sub_timeline, sub_files,
        ) = st.tabs([
            "Surveys", "Task timing", "Task plan", "Ideate", "Debug",
            "Notebook grade", "Quiz", "Verbal assessment", "Session timeline", "Raw files",
        ])

        with sub_surveys:
            render_survey_breakdown(bundle["pre_survey"], "Pre-survey")
            st.divider()
            render_survey_breakdown(bundle["post_survey"], "Post-survey")

        with sub_taskplan:
            if "task_plan" in docs:
                st.markdown(docs["task_plan"]["content"])
            else:
                st.caption("No task plan on file.")
            with st.expander("Upload / replace"):
                _doc_upload_widget(sid, "task_plan", DOC_TITLES["task_plan"], "task_plan" in docs)

        for sub, key, legacy_pair in (
            (sub_ideate, "ideate", ("ideate_manual", "ideate_ai")),
            (sub_debug, "debug", ("debug_manual", "debug_ai")),
        ):
            with sub:
                is_pitch = key == tasks.IDEATE
                if key in docs:
                    # A transcript is line-per-segment plain text; keep its breaks.
                    body = docs[key]["content"]
                    st.markdown(body.replace("\n", "  \n") if is_pitch else body)
                else:
                    st.markdown(
                        "_No pitch transcript uploaded yet._" if is_pitch else "_Not uploaded._"
                    )
                with st.expander("Upload / replace", expanded=is_pitch and key not in docs):
                    if is_pitch:
                        st.caption(
                            "Participants upload nothing for this task — upload the "
                            "transcript of their spoken pitch here (.txt or .md)."
                        )
                    _doc_upload_widget(
                        sid, key, DOC_TITLES[key], key in docs,
                        types=("txt", "md") if is_pitch else ("md",),
                    )
                present_legacy = [k for k in legacy_pair if k in docs]
                if present_legacy:
                    st.divider()
                    st.caption(
                        "This participant was run under the earlier protocol, which "
                        "split this task into a manual and an AI document."
                    )
                    for k in present_legacy:
                        with st.expander(DOC_TITLES[k]):
                            st.markdown(docs[k]["content"])
                            _doc_upload_widget(sid, k, DOC_TITLES[k], True)
                if key in (tasks.DEBUG, tasks.IDEATE):
                    _render_task_grade(sid, key, docs.get(key))

        with sub_timings:
            timings = bundle.get("task_timings") or {}
            if not timings:
                st.caption("No task timings recorded — this participant predates timed tasks.")
            else:
                rows = []
                for key in (tasks.MAIN, tasks.IDEATE, tasks.DEBUG):
                    t = timings.get(key)
                    if not t:
                        continue
                    spent = _timing_seconds(t)
                    limit = t.get("limit_seconds")
                    rows.append({
                        "task": tasks.TASK_TITLE[key],
                        "started": t.get("started_at"),
                        "time spent": tasks.format_duration(spent),
                        "allowance": tasks.format_duration(limit) if limit else "untimed",
                        "over by": (
                            tasks.format_duration(spent - limit)
                            if limit and spent is not None and spent > limit else "—"
                        ),
                        "finished": "yes" if t.get("finished_at") else "still open",
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

            extras = bundle.get("extra_files") or []
            st.markdown("#### Additional main-task files")
            if not extras:
                st.caption("None uploaded.")
            for row in sorted(extras, key=lambda r: r["filename"]):
                c_name, c_dl, c_del = st.columns([3, 1, 1])
                size = row.get("byte_size")
                c_name.markdown(
                    f"📎 `{row['filename']}`"
                    + (f" · {size:,} bytes" if size else "")
                )
                c_dl.download_button(
                    "⬇️", data=extra_file_bytes(row), file_name=row["filename"],
                    key=f"dl_extra_{row['filename']}",
                )
                if c_del.button("🗑️", key=f"del_extra_{row['filename']}"):
                    ok, err = delete_participant_extra_file(sid, row["filename"])
                    (st.success if ok else st.error)("Deleted." if ok else f"Not deleted: {err}")
                    if ok:
                        st.rerun()

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
                st.markdown("#### Graded rubric")
                _render_graded_rubric(nb["results"], sid, "notebook")
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
                _notebook_upload_widget(sid, nb is not None)

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
                    if not notebook_has_content(notebook_text):
                        st.caption(
                            ("The notebook on file is empty (all cells blank)"
                             if notebook_text else "No notebook on file")
                            + " — answers about the notebook will be graded inapplicable; "
                            "conceptual answers are still graded 0-2."
                        )
                    c_btn, c_force = st.columns([2, 2])
                    force_tr = c_force.checkbox("Re-check everything", key="force_reannotate_tr")
                    n_target = len(pairs) if force_tr else n_missing
                    if c_btn.button(
                        f"🏷️ {'Re-check' if force_tr else 'Fact-check'} ({n_target})"
                        if n_target else "🏷️ Fact-check (up to date)",
                        key="annotate_tr", disabled=n_target == 0,
                    ):
                        with st.spinner("Grading answers…"):
                            ok, msg = _annotate_transcript_for(sid, pairs, notebook_text, force=force_tr)
                        (st.success if ok else st.error)(msg)
                        st.rerun()
                # The last four are the pre-0-2-scale labels, still on rows graded
                # earlier; "Re-check everything" replaces them.
                _ACCURACY_ICON = {
                    "fully_correct": "✅", "partially_correct": "⚠️",
                    "incorrect": "❌", "inapplicable": "➖",
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
                        pts = annotate.TRANSCRIPT_SCORES.get(a["accuracy"])
                        score = f" · {pts}/2" if pts is not None else ""
                        st.caption(f"{icon} **{label}**{score} — {a.get('reasoning') or ''}")
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
                raw_by_filename = {lg["filename"]: lg["raw_jsonl"] for lg in logs}
                merged = merge_parsed([parse_jsonl(raw) for raw in raw_by_filename.values()])
                if len(logs) > 1:
                    st.caption(f"Combined across {len(logs)} logs: " + ", ".join(f"`{n}`" for n in raw_by_filename))
                metrics = compute_log_metrics(merged)
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
                render_parsed_timeline(merged, key=sid)

                st.markdown("#### Turn annotations")
                # Per-log parses (original, un-offset turn numbers) interleaved by
                # real time -- annotations are stored keyed by each log's own
                # turn numbering, not merged's collision-safe offset numbering.
                tagged_prompts = []
                for fn, raw in raw_by_filename.items():
                    for e in parse_jsonl(raw)["events"]:
                        if e["lane"] == "User Prompt":
                            tagged_prompts.append((fn, e))
                tagged_prompts.sort(key=lambda pair: pair[1]["t"])

                anno_map = {
                    (a["log_filename"], a["turn_index"]): a
                    for a in list_turn_annotations(sid) if a["source"] == "log"
                }
                n_missing = sum(1 for fn, e in tagged_prompts if (fn, e["turn"]) not in anno_map)
                c_btn, c_force = st.columns([2, 2])
                force_log = c_force.checkbox("Re-annotate everything", key="force_reannotate_log")
                n_target = len(tagged_prompts) if force_log else n_missing
                if c_btn.button(
                    f"🏷️ {'Re-annotate' if force_log else 'Annotate'} ({n_target})"
                    if n_target else "🏷️ Annotate (up to date)",
                    key="annotate_log", disabled=n_target == 0,
                ):
                    with st.spinner(f"Annotating across {len(logs)} log(s)…"):
                        ok, msg = _annotate_all_logs_for(sid, logs, force=force_log)
                    (st.success if ok else st.error)(msg)
                    st.rerun()
                if anno_map:
                    for fn, e in tagged_prompts:
                        a = anno_map.get((fn, e["turn"]))
                        if not a:
                            continue
                        c_lbl, c_tag = st.columns([6, 1])
                        prefix = f"`{fn}` · " if len(logs) > 1 else ""
                        c_lbl.caption(
                            f"{prefix}Turn {e['turn']} · **{a['phase']}** · "
                            f"{a['delegation_posture']} · {a['trust_behavior']}"
                        )
                        with c_tag.popover("💬"):
                            st.write(a.get("turn_text") or "")
                            st.caption(a.get("reasoning") or "")
                            if st.button(
                                "🔄 Re-annotate this turn", key=f"reanno_log_{fn}_{e['turn']}"
                            ):
                                with st.spinner("Re-annotating…"):
                                    ok, msg = _reannotate_one_log_turn(
                                        sid, raw_by_filename[fn], fn, e["turn"]
                                    )
                                (st.success if ok else st.error)(msg)
                                st.rerun()
                else:
                    st.caption("No annotations yet — click Annotate above.")

                st.divider()
                st.markdown("#### Manage logs")
                for lg in logs:
                    c_name, c_conf, c_del = st.columns([4, 2, 1])
                    c_name.write(f"`{lg['filename']}`")
                    confirm = c_conf.checkbox(
                        "Confirm delete", key=f"confirm_del_log_{lg['filename']}"
                    )
                    if c_del.button(
                        "🗑️", key=f"del_log_{lg['filename']}", disabled=not confirm
                    ):
                        ok, err = delete_participant_log(sid, lg["filename"])
                        (st.success if ok else st.error)(
                            f"Deleted `{lg['filename']}`." if ok else f"Not deleted: {err}"
                        )
                        if ok:
                            st.rerun()

            st.divider()
            st.markdown("#### Add log(s)")
            st.caption(
                "Adds session-log file(s) for this participant (e.g. separate work "
                "sessions) — select multiple at once if you have them. Each "
                "uploaded file's own name is used as-is; re-uploading a name "
                "already on file replaces just that log."
            )
            ups = st.file_uploader(
                "Session log(s) (.jsonl)", type=["jsonl"], accept_multiple_files=True, key="up_log"
            )
            if ups and st.button(f"Save {len(ups)} log(s)", key="save_log"):
                results = []
                for up in ups:
                    raw_jsonl = up.getvalue().decode("utf-8", errors="replace")
                    new_metrics = compute_log_metrics(parse_jsonl(raw_jsonl))
                    ok, err = save_participant_log(sid, up.name, raw_jsonl, new_metrics)
                    results.append((up.name, ok, err))
                for name, ok, err in results:
                    (st.success if ok else st.error)(
                        f"{name}: saved" if ok else f"{name}: not saved ({err})"
                    )
                if all(ok for _, ok, _ in results):
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
            for row in sorted(bundle.get("extra_files") or [], key=lambda r: r["filename"]):
                st.download_button(
                    f"⬇️ {row['filename']} (extra)", data=extra_file_bytes(row),
                    file_name=row["filename"], key=f"dlraw_extra_{row['filename']}",
                )
            if not bundle["files"] and not nb and not logs and not bundle.get("extra_files"):
                st.caption("No files on record.")

# ---- Tab 3: Cohort statistics ----------------------------------------
with tab_cohort:
    cohort.render_cohort(list_participant_summaries(), _section_rows)

# ---- Tab 4: Survey results (cohort) ----------------------------------
with tab_surveys:
    survey_stats.render_survey_cohort(
        list_participant_survey_rows(), list_participant_summaries()
    )

# ---- Tab 5: Task instructions ---------------------------------------
with tab_tasks:
    st.markdown(
        "What the participant reads before each task. The main task shows two "
        "documents on one page: a shared task brief (the task itself, the same "
        "for every condition) followed by that participant's condition "
        "instructions (how to go about it) — only the one matching the "
        "condition they were assigned. Ideation and debugging are each one "
        "document, shared across conditions."
    )
    st.caption(
        "Markdown, rendered to the participant exactly as previewed here. A task "
        f"whose document is missing cannot be started, so upload all "
        f"{len(tasks.INSTRUCTION_KEYS)} before running anyone."
    )

    # Saving reruns immediately to refresh the previews, which would wipe the
    # success message before it renders — same flash trick as the Grading tab.
    _tflash = st.session_state.pop("_task_save_msg", None)
    if _tflash is not None:
        (st.success if _tflash[0] else st.error)(_tflash[1])

    _instructions = list_task_instructions()
    _missing = [k for k, _ in tasks.INSTRUCTION_KEYS if k not in _instructions]
    if _missing:
        st.error(
            "Not uploaded yet: "
            + ", ".join(f"**{tasks.INSTRUCTION_TITLE[k]}**" for k in _missing)
        )
    else:
        st.success(f"All {len(tasks.INSTRUCTION_KEYS)} task documents are uploaded.")

    for _key, _title in tasks.INSTRUCTION_KEYS:
        _row = _instructions.get(_key)
        _mark = "✅" if _row else "❌"
        with st.expander(f"{_mark} {_title}", expanded=_row is None):
            if _row:
                st.caption(
                    f"`{_key}` · last saved {_row.get('updated_at') or 'unknown'}"
                )
            else:
                st.caption(f"`{_key}` · nothing uploaded yet")

            # The editor's text lives in session state so "Load into editor" can
            # fill it in for review; a rerun applies that before the widget is
            # created -- a text_area's `value=` kwarg is otherwise ignored once its
            # key already exists in session state, so a plain `value=_seed` here
            # would silently keep showing whatever was already in the box (blank,
            # for a document with nothing saved yet) instead of the uploaded file.
            _body_key = f"ti_body_{_key}"
            _draft = st.session_state.pop(f"_ti_draft_{_key}", None)
            if _draft is not None:
                st.session_state[_body_key] = _draft
            elif _body_key not in st.session_state:
                st.session_state[_body_key] = _row.get("content") if _row else ""

            _up = st.file_uploader(
                "Upload a .md file", type=["md"], key=f"ti_up_{_key}"
            )
            if _up is not None and st.button("Load into editor", key=f"ti_load_{_key}"):
                st.session_state[f"_ti_draft_{_key}"] = _decode_upload(_up)
                st.rerun()
            _heading = st.text_input(
                "Heading shown above the instructions (optional)",
                value=(_row.get("title") or "") if _row else "",
                key=f"ti_title_{_key}",
            )
            _body = st.text_area(
                "Instructions (markdown)", height=280, key=_body_key,
            )
            if _body.strip():
                with st.expander("Preview as the participant sees it"):
                    if _heading.strip():
                        st.markdown(f"### {_heading.strip()}")
                    st.markdown(_body)

            c_save, c_del = st.columns([3, 1])
            if c_save.button("Save", type="primary", key=f"ti_save_{_key}",
                             disabled=not _body.strip()):
                ok, err = save_task_instruction(_key, _body, _heading)
                st.session_state["_task_save_msg"] = (
                    ok, f"Saved {_title}." if ok else f"Not saved: {err}"
                )
                st.rerun()
            if _row and c_del.button("🗑️ Delete", key=f"ti_del_{_key}"):
                ok, err = delete_task_instruction(_key)
                if ok:
                    st.session_state.pop(_body_key, None)
                st.session_state["_task_save_msg"] = (
                    ok, f"Deleted {_title}." if ok else f"Not deleted: {err}"
                )
                st.rerun()

    # -- Debugging task: the buggy notebook, shown to participants + used for autograding ------
    st.divider()
    st.markdown("### 🐞 Debugging task — the buggy notebook")
    st.caption(
        "Upload the buggy notebook participants debug. It's rendered to HTML and "
        "offered to participants as an \"open in a new tab\" link on the Debugging "
        "task screen. Claude also reads it alongside each write-up, finds every bug "
        "the participant reported, and scores each one: +1 if it is a real bug, +1 if "
        "the fix is valid (max 2 per reported bug) -- that grading use stays admin-only."
    )
    _dbg_nb = _instructions.get(tasks.DEBUG_NOTEBOOK_KEY)

    with st.expander(f"{'✅' if _dbg_nb else '❌'} Buggy notebook", expanded=_dbg_nb is None):
        if _dbg_nb:
            st.caption(
                f"`{_dbg_nb.get('title') or 'notebook'}` · last saved "
                f"{_dbg_nb.get('updated_at') or 'unknown'}. Replacing it does not regrade "
                "anyone — use Re-grade on each participant."
            )
        _dbg_up = st.file_uploader("Upload the buggy notebook (.ipynb)", type=["ipynb"], key="dbg_nb_up")
        if _dbg_up is not None and st.button("Save notebook", type="primary", key="dbg_nb_save"):
            _dbg_bytes = _dbg_up.getvalue()
            try:
                _dbg_text = notebook_to_text(_dbg_bytes)
            except Exception as e:
                st.session_state["_task_save_msg"] = (False, f"Could not read notebook: {e}")
                st.rerun()
            if not notebook_has_content(_dbg_text):
                st.session_state["_task_save_msg"] = (False, "That notebook has no content — every cell is blank.")
            else:
                try:
                    _dbg_html = notebook_to_html(_dbg_bytes)
                except Exception as e:
                    st.session_state["_task_save_msg"] = (False, f"Could not render notebook to HTML: {e}")
                    st.rerun()
                ok, err = save_task_instruction(
                    tasks.DEBUG_NOTEBOOK_KEY, _dbg_text, _dbg_up.name, notebook_html=_dbg_html
                )
                st.session_state["_task_save_msg"] = (
                    ok, "Saved the buggy notebook." if ok else f"Not saved: {err}"
                )
            st.rerun()
        if _dbg_nb:
            with st.expander("Notebook as the grader sees it (flattened, with cell numbers)"):
                st.code(_dbg_nb["content"])
            if _dbg_nb.get("notebook_html"):
                st.caption("✅ Rendered HTML is saved — participants can open it from the Debugging task screen.")
            else:
                st.caption("⚠️ No rendered HTML saved yet — re-upload the notebook to generate it.")
            if st.button("🗑️ Delete notebook", key="dbg_nb_del"):
                ok, err = delete_task_instruction(tasks.DEBUG_NOTEBOOK_KEY)
                st.session_state["_task_save_msg"] = (
                    ok, "Deleted the buggy notebook." if ok else f"Not deleted: {err}"
                )
                st.rerun()

    # -- Ideation task: the rubric its pitch transcripts are graded against ------
    st.divider()
    st.markdown("### 💡 Idea generation task — grading rubric")
    st.caption(
        "Admin only. Participants upload nothing for this task: they pitch their idea "
        "aloud and you upload the transcript under Participant → Ideate. It is scored "
        "against this rubric, which the grader reads exactly as written (any format — "
        "a CSV or a markdown table both work — as long as it states each criterion's "
        "points and what earns them)."
    )
    _ide_rubric = _instructions.get(tasks.IDEATE_RUBRIC_KEY)
    with st.expander(
        f"{'✅' if _ide_rubric else '❌'} Ideation rubric", expanded=_ide_rubric is None
    ):
        _ide_draft = st.session_state.pop("_ide_rubric_draft", None)
        if _ide_draft is not None:
            st.session_state["ide_rubric_body"] = _ide_draft
        elif "ide_rubric_body" not in st.session_state:
            st.session_state["ide_rubric_body"] = _ide_rubric["content"] if _ide_rubric else ""
        if _ide_rubric:
            st.caption(
                f"Last saved {_ide_rubric.get('updated_at') or 'unknown'}. Saving does not "
                "regrade anyone — use Re-grade on each participant."
            )
        _ide_up = st.file_uploader("Load a rubric file", type=["csv", "md", "txt"], key="ide_rubric_up")
        if _ide_up is not None and st.button("Load into editor", key="ide_rubric_load"):
            st.session_state["_ide_rubric_draft"] = _decode_upload(_ide_up)
            st.rerun()
        _ide_body = st.text_area("Rubric", height=300, key="ide_rubric_body")
        _c_isave, _c_idel = st.columns([3, 1])
        if _c_isave.button(
            "Save rubric", type="primary", key="ide_rubric_save", disabled=not _ide_body.strip()
        ):
            ok, err = save_task_instruction(
                tasks.IDEATE_RUBRIC_KEY, _ide_body, "Ideation rubric"
            )
            st.session_state["_task_save_msg"] = (
                ok, "Saved the ideation rubric." if ok else f"Not saved: {err}"
            )
            st.rerun()
        if _ide_rubric and _c_idel.button("🗑️ Delete", key="ide_rubric_del"):
            ok, err = delete_task_instruction(tasks.IDEATE_RUBRIC_KEY)
            st.session_state.pop("ide_rubric_body", None)
            st.session_state["_task_save_msg"] = (
                ok, "Deleted the ideation rubric." if ok else f"Not deleted: {err}"
            )
            st.rerun()

# ---- Tab 6: Notebook grading report --------------------------------
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

# ---- Tab 7: Grading & rubric ----------------------------------------
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
            with st.expander("Grading instructions (the \"how to grade\" prompt sent to Opus 5)"):
                st.markdown(viewed.get("grading_instructions") or grading.DEFAULT_GRADING_INSTRUCTIONS)
                if not viewed.get("grading_instructions"):
                    st.caption("(using the built-in default — not customized for this rubric)")
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
        "Task description (graded against)", value=task_default, height=220,
        key=f"rubric_task_{name}",
    )

    st.markdown("###### Grading instructions")
    st.caption(
        "The \"how to grade\" prompt sent directly to Opus 5 — scoring scale, what "
        "counts as evidence, etc. Separate from the task description above."
    )
    instructions_default = (target or {}).get("grading_instructions") or grading.DEFAULT_GRADING_INSTRUCTIONS
    _instr_key = f"rubric_instructions_{name}"
    instructions_md = st.file_uploader(
        "Upload the grading instructions as a .md file — fills in the box below",
        type=["md"], key=f"instructions_md_up_{name}",
    )
    if instructions_md is not None:
        # Only overwrite on a genuinely new upload, not every rerun -- otherwise
        # this would stomp on a manual edit made to the box afterward.
        _applied_key = f"_instructions_md_applied_{name}"
        if st.session_state.get(_applied_key) != instructions_md.name:
            st.session_state[_instr_key] = _decode_upload(instructions_md)
            st.session_state[_applied_key] = instructions_md.name
    instructions_text = st.text_area(
        "Grading instructions", value=instructions_default, height=220, key=_instr_key,
        label_visibility="collapsed",
    )

    up = st.file_uploader("Rubric CSV", type=["csv"])
    if up is not None and st.button("Save rubric", type="primary"):
        try:
            csv_text = up.getvalue().decode("utf-8-sig")
        except UnicodeDecodeError:
            csv_text = up.getvalue().decode("latin-1")
        ok, err = save_rubric(name, task_text, csv_text, instructions_text)
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
    for row in list_participant_transcripts_raw():
        pairs = row.get("parsed") or []
        if not pairs:
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
