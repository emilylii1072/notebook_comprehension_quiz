"""Participant intake — the only participant-facing page.

Flow: identify (subject ID + condition) → upload the 7 study files → take the
comprehension quiz → done. The notebook is graded synchronously during submission,
but **no score, breakdown, or grading error is ever shown to the participant** — all
of that lives behind the admin gate.

One page of the multipage app — run via `streamlit run app.py`.
Auth: set ANTHROPIC_API_KEY (quiz generation + hidden grading).
"""

import re

import streamlit as st
from dotenv import load_dotenv

from db import (
    get_rubric,
    get_secret,
    mark_participant_complete,
    participant_exists,
    save_participant_file,
    save_participant_log,
    save_participant_notebook,
    set_participant_grading_error,
    update_participant_grading,
    upsert_participant,
)
from lib import grading
from lib.llm import get_client
from lib.notebook import notebook_to_text
from lib.quiz_ui import render_quiz_flow
from lib.timeline import compute_log_metrics, parse_jsonl

load_dotenv()

SUBJECT_ID_RE = re.compile(r"^P\d{3}$")
ACTIVE_RUBRIC_NAME = get_secret("ACTIVE_RUBRIC_NAME", grading.DEFAULT_RUBRIC_NAME)

CONDITION_ALIASES = {
    "slow planning": "slow_planning",
    "slow-planning": "slow_planning",
    "slowplanning": "slow_planning",
    "slow iterating": "slow_iterating",
    "slow-iterating": "slow_iterating",
    "slowiterating": "slow_iterating",
    "slow iteration": "slow_iterating",
    "control": "control",
}
CONDITION_LABEL = {
    "slow_planning": "Slow planning",
    "slow_iterating": "Slow iterating",
    "control": "Control",
}

# doc_type -> (extension, is_markdown_doc)
DOC_SUFFIXES = {
    "task_plan": "task_plan.md",
    "debug_manual": "debug_manual.md",
    "debug_ai": "debug_ai.md",
    "ideate_manual": "ideate_manual.md",
    "ideate_ai": "ideate_ai.md",
}


def normalize_condition(text: str) -> str | None:
    key = re.sub(r"[\s_-]+", " ", (text or "").strip().lower()).strip()
    return CONDITION_ALIASES.get(key) or CONDITION_ALIASES.get(key.replace(" ", ""))


def expected_files(subject_id: str) -> dict[str, tuple[str, str | None]]:
    """{exact filename: (kind, doc_type)} for the 7 required uploads."""
    exp: dict[str, tuple[str, str | None]] = {}
    for doc_type, suffix in DOC_SUFFIXES.items():
        exp[f"{subject_id}_{suffix}"] = ("doc", doc_type)
    exp[f"{subject_id}_notebook.ipynb"] = ("notebook", None)
    exp[f"{subject_id}_claude_log.jsonl"] = ("log", None)
    return exp


def reset() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.p_stage = "identify"


if "p_stage" not in st.session_state:
    st.session_state.p_stage = "identify"

st.title("📥 Study Submission")

# ---- Stage: identify ---------------------------------------------------------
if st.session_state.p_stage == "identify":
    st.markdown(
        "Enter your **Subject ID** and the **condition** you were assigned in person, "
        "then upload your study files and complete a short comprehension quiz."
    )
    subject_id = st.text_input("Subject ID", placeholder="P001").strip()
    condition_text = st.text_input(
        "Condition (as assigned to you)", placeholder="e.g. control"
    ).strip()

    id_ok = bool(SUBJECT_ID_RE.match(subject_id))
    condition = normalize_condition(condition_text)

    if subject_id and not id_ok:
        st.error("Subject ID must be the letter P followed by 3 digits (e.g. P001).")
    if condition_text and condition is None:
        st.error("Unrecognized condition. Enter the exact condition you were assigned.")

    overwrite_ok = True
    if id_ok and participant_exists(subject_id):
        st.warning(
            f"A submission for **{subject_id}** already exists. Continuing will "
            "replace it entirely."
        )
        overwrite_ok = st.checkbox("Replace the existing submission")

    if st.button("Continue", type="primary", disabled=not (id_ok and condition and overwrite_ok)):
        ok, err = upsert_participant(subject_id, condition)
        if not ok:
            st.error(f"Could not start the submission: {err}")
            st.stop()
        st.session_state.subject_id = subject_id
        st.session_state.condition = condition
        st.session_state.p_stage = "upload"
        st.rerun()

# ---- Stage: upload ---------------------------------------------------------
elif st.session_state.p_stage == "upload":
    subject_id = st.session_state.subject_id
    exp = expected_files(subject_id)

    st.markdown(
        f"**Subject {subject_id}** · condition recorded.\n\n"
        "Select **all 7 files** from your notebook folder. Every file must be named "
        f"`{subject_id}_<name>` exactly:"
    )
    st.code("\n".join(sorted(exp)), language="text")

    uploaded = st.file_uploader(
        "Study files", type=["md", "ipynb", "jsonl"], accept_multiple_files=True
    )
    by_name = {f.name: f for f in (uploaded or [])}

    missing = [name for name in exp if name not in by_name]
    unexpected = [name for name in by_name if name not in exp]

    st.markdown("#### File check")
    for name in sorted(exp):
        st.markdown(("✅ " if name in by_name else "❌ ") + f"`{name}`")
    for name in sorted(unexpected):
        st.markdown(f"🚫 unexpected: `{name}` — remove or rename this file")

    ready = not missing and not unexpected
    if not ready:
        st.info("Submit unlocks once exactly the 7 correctly-named files are selected.")

    if st.button("Submit files", type="primary", disabled=not ready):
        errors: list[str] = []

        # Notebook — parse before persisting anything so an unreadable notebook stops here.
        nb_name = f"{subject_id}_notebook.ipynb"
        try:
            notebook_text = notebook_to_text(by_name[nb_name].getvalue())
        except Exception as e:
            st.error(f"Could not read {nb_name}: {e}")
            st.stop()
        if not notebook_text.strip():
            st.error(f"{nb_name} appears to be empty.")
            st.stop()

        with st.spinner("Saving your files…"):
            for name, (kind, doc_type) in exp.items():
                if kind != "doc":
                    continue
                try:
                    content = by_name[name].getvalue().decode("utf-8-sig")
                except UnicodeDecodeError:
                    content = by_name[name].getvalue().decode("latin-1")
                ok, err = save_participant_file(subject_id, doc_type, name, content)
                if not ok:
                    errors.append(f"{name}: {err}")

            ok, err = save_participant_notebook(subject_id, nb_name, notebook_text)
            if not ok:
                errors.append(f"{nb_name}: {err}")

            log_name = f"{subject_id}_claude_log.jsonl"
            raw_jsonl = by_name[log_name].getvalue().decode("utf-8", errors="replace")
            metrics = compute_log_metrics(parse_jsonl(raw_jsonl))
            ok, err = save_participant_log(subject_id, log_name, raw_jsonl, metrics)
            if not ok:
                errors.append(f"{log_name}: {err}")

        if errors:
            st.error("Some files could not be saved:\n\n" + "\n".join(errors))
            st.stop()

        st.session_state.notebook_text = notebook_text
        st.session_state.notebook_filename = nb_name
        st.session_state.file_manifest = {"expected": sorted(exp), "received": sorted(by_name)}
        st.session_state.p_stage = "quiz"
        st.rerun()

# ---- Stage: quiz ---------------------------------------------------------
elif st.session_state.p_stage == "quiz":
    st.markdown(
        "Answer each question about **your own submission**. There is no time limit — "
        "a stopwatch just tracks how long you take. You won't see a score."
    )
    done = render_quiz_flow(
        notebook_text=st.session_state.notebook_text,
        notebook_filename=st.session_state.notebook_filename,
        subject_id=st.session_state.subject_id,
    )
    if done:
        st.session_state.p_stage = "finalize"
        st.rerun()

# ---- Stage: finalize (hidden-synchronous grading) --------------------------
elif st.session_state.p_stage == "finalize":
    subject_id = st.session_state.subject_id
    with st.spinner("Finalizing your submission…"):
        rubric = get_rubric(ACTIVE_RUBRIC_NAME)
        if rubric is None or not (rubric.get("rubric_csv") or "").strip():
            set_participant_grading_error(
                subject_id, f"No usable rubric '{ACTIVE_RUBRIC_NAME}' configured."
            )
        else:
            try:
                results = grading.grade_notebook(
                    get_client(), rubric["task"], rubric["rubric_csv"],
                    st.session_state.notebook_text,
                )
                total = round(sum(r["score"] for r in results), 2)
                mx = round(sum(r["max_pts"] for r in results), 2)
                ok, err = update_participant_grading(
                    subject_id, ACTIVE_RUBRIC_NAME, grading.MODEL, results, total, mx
                )
                if not ok:
                    set_participant_grading_error(subject_id, err or "grading save failed")
            except Exception as e:  # never surface a grading failure to the participant
                set_participant_grading_error(subject_id, f"{type(e).__name__}: {e}")

        mark_participant_complete(subject_id, st.session_state.get("file_manifest") or {})

    st.session_state.p_stage = "done"
    st.rerun()

# ---- Stage: done ---------------------------------------------------------
elif st.session_state.p_stage == "done":
    st.success(
        f"✅ Submission received for **{st.session_state.subject_id}** — 7 files and "
        "the quiz are complete. Thank you!"
    )
    st.caption("You can close this tab. Nothing further is required.")
    st.button("Start another submission", on_click=reset)
