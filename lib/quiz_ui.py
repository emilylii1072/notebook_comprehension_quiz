"""The in-app quiz experience.

`render_quiz_flow` runs the participant-facing quiz: generate → answer one question
at a time (with a stopwatch and two live follow-ups) → save to the database. It NEVER
shows a score or a per-question breakdown — the participant only learns that the quiz
is complete.

`render_quiz_breakdown` renders a completed quiz record (score + per-question detail)
read-only, for the admin's participant view.

Lifted from the old Notebook Quiz page's `quiz` and `results` stages.
"""

import time

import streamlit as st

from db import save_participant_quiz
from lib.llm import get_client
from lib.quiz import (
    FOLLOWUP_AFTER,
    FOLLOWUP_SUBJECT_FOR_TRIGGER,
    QuizQuestion,
    assemble_static_questions,
    call_with_errors_surfaced,
    generate_label_followup,
    generate_model_followup,
)


def build_items(static_questions: list[QuizQuestion]) -> list:
    """Interleave the static questions with follow-up placeholders, producing the
    full quiz sequence in template order."""
    items = []
    for q in static_questions:
        items.append(q)
        if q.slot in FOLLOWUP_AFTER:
            items.append({"placeholder": True, "subject": FOLLOWUP_SUBJECT_FOR_TRIGGER[q.slot]})
    return items


def _skipped_question(slot: str, reason: str) -> QuizQuestion:
    """A trivial placeholder for a follow-up that won't be generated. Excluded from
    scoring via the `topic == "skipped"` check in _finish_quiz."""
    return QuizQuestion(
        slot=slot, topic="skipped",
        question=f"_({reason})_",
        options=["N/A", "N/A", "N/A", "N/A"], correct_index=0,
        explanation=reason,
    )


def _record_current_answer() -> None:
    """Save the radio selection (and elapsed time) for the current question index."""
    ss = st.session_state
    idx = ss.qz_current_index
    item = ss.qz_items[idx]
    # Streamlit re-defines QuizQuestion on every rerun, so an instance stored earlier
    # is never `isinstance` of this rerun's class. Check the stable builtin `dict`
    # (the placeholder marker) instead.
    if not isinstance(item, dict):
        selected = ss.get(f"qz_q{idx}")
        ss.qz_answers[idx] = (
            item.options.index(selected) if selected in item.options else None
        )
        started = ss.qz_question_start_times.get(idx, time.time())
        ss.qz_time_spent[idx] = round(time.time() - started, 1)


def _finish_quiz(subject_id: str, notebook_filename: str) -> None:
    """Collect materialized questions, score, persist to `participant_quiz`, and mark
    the flow done. The participant is not shown the score."""
    ss = st.session_state
    final_q, final_a, final_t = [], [], []
    for it, a, t in zip(ss.qz_items, ss.qz_answers, ss.qz_time_spent):
        if not isinstance(it, dict) and it.topic != "skipped":
            final_q.append(it)
            final_a.append(a)
            final_t.append(t)

    elapsed = time.time() - ss.qz_started_at
    score = sum(1 for q, a in zip(final_q, final_a) if a == q.correct_index)
    total = len(final_q)

    question_records = [
        {
            "slot": q.slot,
            "topic": q.topic,
            "question": q.question,
            "options": q.options,
            "correct_index": q.correct_index,
            "candidate_answer_index": a,
            "answered_correctly": a == q.correct_index,
            "explanation": q.explanation,
            "time_spent_seconds": t,
        }
        for q, a, t in zip(final_q, final_a, final_t)
    ]
    payload = {
        "notebook_filename": notebook_filename,
        "score": score,
        "total": total,
        "elapsed_seconds": round(elapsed, 1),
        "questions": question_records,
        "generation_warnings": ss.get("generation_warnings") or [],
    }
    ok, err = save_participant_quiz(subject_id, payload)
    ss.qz_save_ok = ok
    ss.qz_save_error = err
    ss.qz_stage = "done"


def _materialize_followup(idx: int, notebook_text: str) -> QuizQuestion | None:
    """Turn a follow-up placeholder at `idx` into a real question based on the
    participant's answer to the preceding trigger question. Returns the question, or
    None if generation failed and the caller should show retry/skip controls."""
    ss = st.session_state
    item = ss.qz_items[idx]
    subject = item["subject"]
    trigger_item = ss.qz_items[idx - 1]
    trigger_answer_idx = ss.qz_answers[idx - 1]
    trigger_is_dict = isinstance(trigger_item, dict)
    chosen_label = (
        trigger_item.options[trigger_answer_idx]
        if not trigger_is_dict and trigger_answer_idx is not None
        else None
    )

    if subject == "label" and (
        trigger_is_dict
        or trigger_answer_idx is None
        or trigger_answer_idx != trigger_item.correct_index
    ):
        ss.qz_items[idx] = _skipped_question(
            "label_followup",
            'Skipped — only asked when the previous question is answered "No".',
        )
        return ss.qz_items[idx]

    with st.spinner("Generating a follow-up based on your answer…"):
        if subject == "label":
            followup = call_with_errors_surfaced(
                generate_label_followup, get_client(), notebook_text,
            )
        else:  # "model"
            model_name = chosen_label or trigger_item.options[trigger_item.correct_index]
            followup = call_with_errors_surfaced(
                generate_model_followup, get_client(), notebook_text,
                model_name, chosen_label is not None,
            )
    if followup is None:
        return None
    ss.qz_items[idx] = followup
    return followup


def render_quiz_flow(*, notebook_text: str, notebook_filename: str, subject_id: str) -> bool:
    """Run the whole quiz. Returns True once it is complete and saved."""
    ss = st.session_state
    if "qz_stage" not in ss:
        ss.qz_stage = "generate"

    # ---- generate --------------------------------------------------------
    if ss.qz_stage == "generate":
        with st.spinner(
            "Reading the notebook, writing questions, and self-checking each one… "
            "(~2 minutes)"
        ):
            static_questions = call_with_errors_surfaced(
                assemble_static_questions, get_client(), notebook_text
            )
        if static_questions is None:
            if st.button("Retry quiz generation"):
                st.rerun()
            st.stop()
        items = build_items(static_questions)
        ss.qz_items = items
        ss.qz_answers = [None] * len(items)
        ss.qz_time_spent = [None] * len(items)
        ss.qz_question_start_times = {}
        ss.qz_current_index = 0
        ss.qz_started_at = time.time()
        ss.qz_stage = "quiz"
        st.rerun()

    # ---- done -----------------------------------------------------------
    if ss.qz_stage == "done":
        return True

    # ---- quiz ---------------------------------------------------------
    idx = ss.qz_current_index
    item = ss.qz_items[idx]

    if isinstance(item, dict) and item.get("placeholder"):
        materialized = _materialize_followup(idx, notebook_text)
        if materialized is None:
            col_a, col_b = st.columns(2)
            if col_a.button("Retry"):
                st.rerun()
            if col_b.button("Skip this question"):
                ss.qz_items[idx] = _skipped_question(
                    f"{item['subject']}_followup",
                    "This follow-up could not be generated and was skipped.",
                )
                ss.qz_current_index += 1
                st.rerun()
            st.stop()
        item = materialized

    if idx not in ss.qz_question_start_times:
        ss.qz_question_start_times[idx] = time.time()

    @st.fragment(run_every=1.0)
    def stopwatch():
        elapsed = time.time() - ss.qz_started_at
        mm, sec = divmod(int(elapsed), 60)
        st.metric("⏱️ Time elapsed", f"{mm}:{sec:02d}")

    stopwatch()
    st.divider()

    st.subheader(f"Question {idx + 1} of {len(ss.qz_items)}")
    st.markdown(item.question)
    st.radio(
        "Select one:", item.options, index=None,
        key=f"qz_q{idx}", label_visibility="collapsed",
    )

    is_last = idx == len(ss.qz_items) - 1
    if st.button("Submit answers" if is_last else "Next question", type="primary"):
        _record_current_answer()
        if is_last:
            _finish_quiz(subject_id, notebook_filename)
        else:
            ss.qz_current_index += 1
        st.rerun()

    return False


def render_quiz_breakdown(quiz_row: dict) -> None:
    """Render a saved quiz record (score + per-question detail), read-only."""
    score = quiz_row.get("score", 0)
    total = quiz_row.get("total", 0)
    elapsed = quiz_row.get("elapsed_seconds")
    questions = quiz_row.get("questions") or []

    col1, col2 = st.columns(2)
    col1.metric("Score", f"{score} / {total}")
    col2.metric("Time used", f"{elapsed:.0f}s" if elapsed is not None else "—")

    for i, q in enumerate(questions):
        a = q.get("candidate_answer_index")
        correct_idx = q.get("correct_index")
        options = q.get("options") or []
        correct = a == correct_idx
        icon = "✅" if correct else ("⬜" if a is None else "❌")
        with st.expander(f"{icon} Q{i + 1} · {q.get('topic', '')}"):
            st.markdown(q.get("question", ""))
            your = options[a] if (a is not None and a < len(options)) else "_unanswered_"
            corr = options[correct_idx] if (correct_idx is not None and correct_idx < len(options)) else "—"
            st.markdown(f"- **Their answer:** {your}")
            st.markdown(f"- **Correct answer:** {corr}")
            st.markdown(f"- **Why:** {q.get('explanation', '')}")
            t = q.get("time_spent_seconds")
            st.markdown(f"- **Time spent:** {t:.0f}s" if t is not None else "- **Time spent:** _unknown_")

    warnings = quiz_row.get("generation_warnings") or []
    if warnings:
        st.caption(f"⚠️ Self-check could not fully resolve {len(warnings)} question(s): " + "; ".join(warnings))
