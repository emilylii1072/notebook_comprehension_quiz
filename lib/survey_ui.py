"""The in-app pre-/post-survey experience: one item per screen, a stopwatch,
per-item elapsed time recorded, saved once at the end. Every question is
required unless `SurveyItem.optional` is set (mirrors force=OFF in the
source .qsf) -- `_missing_answer` keeps the Next/Submit button disabled, and
says what is still outstanding, until the item is complete, but always lets
an optional item through unanswered. Mirrors
lib/quiz_ui.py's flow (`_record_current_answer` / question-start-time
tracking), generalized to the handful of question kinds lib/surveys.py uses
instead of quiz.py's single multiple-choice kind -- there's no LLM generation
here at all, the content is fixed.

Session-state keys are namespaced by `survey_type` ("pre"/"post") so running
both in the same browser session (pre now, post much later) never collides.
"""

import time

import streamlit as st

from db import save_participant_survey
from lib.surveys import ANSWER_KEY, SurveyItem, is_correct


def _k(survey_type: str, name: str) -> str:
    return f"sv_{survey_type}_{name}"


def _missing_answer(item: SurveyItem, ss, a_key: str) -> str | None:
    """Why this item isn't answered yet, or None if it is (including every
    optional item, unconditionally). A matrix needs every row rated, and
    picking a free-text option ("Other", "Student", ...) also requires filling
    the box it reveals."""
    other_needed = "Please fill in the text box."
    if item.kind == "notice" or item.optional:
        return None
    if item.kind in ("short_text", "long_text"):
        return None if (ss.get(a_key) or "").strip() else "An answer is required."
    if item.kind == "single_select":
        selected = ss.get(a_key)
        if selected is None:
            return "Select an option to continue."
        if selected in item.free_text_options:
            return None if (ss.get(f"{a_key}_other") or "").strip() else other_needed
        return None
    if item.kind == "multi_select":
        chosen = [opt for opt in (item.options or []) if ss.get(f"{a_key}_{opt}")]
        if not chosen:
            return "Select at least one option to continue."
        if any(opt in item.free_text_options for opt in chosen):
            return None if (ss.get(f"{a_key}_other") or "").strip() else other_needed
        return None
    if item.kind == "matrix":
        blank = sum(1 for i in range(len(item.rows or [])) if ss.get(f"{a_key}_row{i}") is None)
        return f"{blank} of {len(item.rows or [])} rows still need a rating." if blank else None
    return None


def _record_current_answer(survey_type: str) -> None:
    """Save the answer (and elapsed time) for the current item index."""
    ss = st.session_state
    idx = ss[_k(survey_type, "current_index")]
    items: list[SurveyItem] = ss[_k(survey_type, "items")]
    item = items[idx]
    answers = ss[_k(survey_type, "answers")]
    a_key = f"{_k(survey_type, 'a')}{idx}"

    if item.kind == "notice":
        answers[idx] = None
    elif item.kind in ("short_text", "long_text"):
        answers[idx] = ss.get(a_key, "")
    elif item.kind == "single_select":
        selected = ss.get(a_key)
        other = (
            ss.get(f"{a_key}_other", "")
            if selected in item.free_text_options
            else None
        )
        answers[idx] = {"selected": selected, "other_text": other}
    elif item.kind == "multi_select":
        selected = [opt for opt in (item.options or []) if ss.get(f"{a_key}_{opt}")]
        other = (
            ss.get(f"{a_key}_other", "")
            if any(opt in item.free_text_options for opt in selected)
            else None
        )
        answers[idx] = {"selected": selected, "other_text": other}
    elif item.kind == "matrix":
        answers[idx] = {
            row: ss.get(f"{a_key}_row{i}") for i, row in enumerate(item.rows or [])
        }

    start_times = ss[_k(survey_type, "start_times")]
    started = start_times.get(idx, time.time())
    ss[_k(survey_type, "time_spent")][idx] = round(time.time() - started, 1)


def _finish_survey(subject_id: str, survey_type: str) -> None:
    ss = st.session_state
    items: list[SurveyItem] = ss[_k(survey_type, "items")]
    answers = ss[_k(survey_type, "answers")]
    time_spent = ss[_k(survey_type, "time_spent")]

    responses = [
        {
            "item_id": it.id,
            "category": it.category,
            "question": it.question,
            "answer": a,
            "time_spent_seconds": t,
        }
        for it, a, t in zip(items, answers, time_spent)
        if it.kind != "notice"
    ]
    elapsed = round(time.time() - ss[_k(survey_type, "started_at")], 1)
    ok, err = save_participant_survey(subject_id, survey_type, responses, elapsed)
    ss[_k(survey_type, "save_ok")] = ok
    ss[_k(survey_type, "save_error")] = err
    ss[_k(survey_type, "stage")] = "done"


def render_survey_flow(*, subject_id: str, survey_type: str, items: list[SurveyItem]) -> bool:
    """Run one survey (pre or post) end to end. Returns True once saved."""
    ss = st.session_state
    stage_key = _k(survey_type, "stage")
    if stage_key not in ss:
        ss[_k(survey_type, "items")] = items
        ss[_k(survey_type, "answers")] = [None] * len(items)
        ss[_k(survey_type, "time_spent")] = [None] * len(items)
        ss[_k(survey_type, "start_times")] = {}
        ss[_k(survey_type, "current_index")] = 0
        ss[_k(survey_type, "started_at")] = time.time()
        ss[stage_key] = "survey"

    if ss[stage_key] == "done":
        if ss.get(_k(survey_type, "save_ok")) is False:
            st.error(f"Could not save: {ss.get(_k(survey_type, 'save_error'))}")
            if st.button("Retry save", key=_k(survey_type, "retry_save")):
                _finish_survey(subject_id, survey_type)
                st.rerun()
            return False
        return True

    idx = ss[_k(survey_type, "current_index")]
    item = items[idx]
    start_times = ss[_k(survey_type, "start_times")]
    if idx not in start_times:
        start_times[idx] = time.time()

    if item.kind != "notice":
        st.caption(
            f"Question {idx + 1} of {len(items)}"
            + (f" · _{item.category}_" if item.category else "")
            + (" · optional" if item.optional else "")
        )
    st.markdown(item.question)

    a_key = f"{_k(survey_type, 'a')}{idx}"
    if item.kind == "notice":
        pass
    elif item.kind == "short_text":
        st.text_input("Your answer", key=a_key, label_visibility="collapsed")
    elif item.kind == "long_text":
        st.text_area("Your answer", key=a_key, label_visibility="collapsed")
    elif item.kind == "single_select":
        st.radio(
            "Select one:", item.options or [], index=None, key=a_key,
            label_visibility="collapsed",
        )
        if ss.get(a_key) in item.free_text_options:
            st.text_input("Please specify:", key=f"{a_key}_other")
    elif item.kind == "multi_select":
        for opt in item.options or []:
            st.checkbox(opt, key=f"{a_key}_{opt}")
        if any(ss.get(f"{a_key}_{opt}") for opt in item.free_text_options):
            st.text_input("Please specify:", key=f"{a_key}_other")
    elif item.kind == "matrix":
        for i, row in enumerate(item.rows or []):
            st.radio(row, item.options or [], index=None, key=f"{a_key}_row{i}", horizontal=True)

    missing = _missing_answer(item, ss, a_key)
    if missing:
        st.caption(f":orange[{missing}]")

    is_last = idx == len(items) - 1
    label = "Submit" if is_last else ("Continue" if item.kind == "notice" else "Next")
    if st.button(
        label, type="primary", key=f"{_k(survey_type, 'next')}{idx}",
        disabled=bool(missing),
    ):
        _record_current_answer(survey_type)
        if is_last:
            _finish_survey(subject_id, survey_type)
        else:
            ss[_k(survey_type, "current_index")] += 1
        st.rerun()

    return False


def render_survey_breakdown(survey_row: dict | None, label: str) -> None:
    """Render a saved survey record (every item, its answer, and its per-item
    timing), read-only, for the admin detail view. Knowledge-assessment items
    are marked against lib.surveys.ANSWER_KEY and totalled at the top; every
    other item is opinion/experience and carries no verdict."""
    if not survey_row:
        st.caption(f"{label}: not yet taken.")
        return
    responses = survey_row.get("responses") or []
    elapsed = survey_row.get("elapsed_seconds")
    verdicts = {
        r.get("item_id"): is_correct(r.get("item_id"), r.get("answer")) for r in responses
    }
    scored = [v for v in verdicts.values() if v is not None]

    c1, c2 = st.columns(2)
    c1.metric(f"{label} — total time", f"{elapsed:.0f}s" if elapsed is not None else "—")
    c2.metric(
        f"{label} — knowledge score",
        f"{sum(scored)}/{len(scored)}" if scored else "—",
        help="Scored items only, against lib.surveys.ANSWER_KEY.",
    )

    for i, r in enumerate(responses):
        cat = f" · _{r['category']}_" if r.get("category") else ""
        t = r.get("time_spent_seconds")
        verdict = verdicts.get(r.get("item_id"))
        mark = {True: "✅ ", False: "❌ ", None: ""}[verdict]
        title = f"{mark}Q{i + 1}{cat}" + (f" — {t:.0f}s" if t is not None else "")
        with st.expander(title):
            st.markdown(r.get("question", ""))
            answer = r.get("answer")
            if isinstance(answer, dict) and "selected" in answer:
                sel = answer["selected"]
                sel_str = ", ".join(sel) if isinstance(sel, list) else (sel or "_no answer_")
                other = answer.get("other_text")
                st.markdown(f"**Answer:** {sel_str}" + (f" — {other}" if other else ""))
            elif isinstance(answer, dict):  # matrix: {row: value}
                for row, val in answer.items():
                    st.markdown(f"- **{row}:** {val or '_no answer_'}")
            else:  # short_text / long_text
                st.markdown(f"**Answer:** {answer if answer else '_no answer_'}")
            if verdict is False:
                st.caption(f"Correct answer: {ANSWER_KEY[r['item_id']]}")
