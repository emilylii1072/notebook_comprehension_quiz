"""Participant intake — the only participant-facing page.

Two-part flow, both keyed to the same subject ID:

  Part 1 (right after the coding task): enter subject ID + condition, upload just
    the notebook, take the comprehension quiz. The notebook is graded synchronously
    behind the scenes. The notebook and quiz result are stored.
  Part 2 (any time later): re-enter the subject ID, upload the five reflection
    documents and the Claude Code session log. The notebook is NOT re-uploaded.

No score, breakdown, or grading error is ever shown to the participant — all of
that lives behind the admin gate.

One page of the multipage app — run via `streamlit run app.py`.
Auth: set ANTHROPIC_API_KEY (quiz generation + hidden grading).
"""

import re

import streamlit as st
from dotenv import load_dotenv

from db import (
    get_participant,
    get_rubric,
    get_secret,
    mark_participant_complete,
    mark_participant_stage1_done,
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

# doc_type -> filename suffix (the five Part 2 markdown documents)
DOC_SUFFIXES = {
    "task_plan": "task_plan.md",
    "debug_manual": "debug_manual.md",
    "debug_ai": "debug_ai.md",
    "ideate_manual": "ideate_manual.md",
    "ideate_ai": "ideate_ai.md",
}
DOC_TITLES = {
    "task_plan": "Task plan",
    "debug_manual": "Debug — manual", "debug_ai": "Debug — AI",
    "ideate_manual": "Ideate — manual", "ideate_ai": "Ideate — AI",
}


def normalize_condition(text: str) -> str | None:
    key = re.sub(r"[\s_-]+", " ", (text or "").strip().lower()).strip()
    return CONDITION_ALIASES.get(key) or CONDITION_ALIASES.get(key.replace(" ", ""))


def notebook_filename(subject_id: str) -> str:
    return f"{subject_id}_notebook.ipynb"


def stage2_expected(subject_id: str) -> dict[str, tuple[str, str | None]]:
    """{exact filename: (kind, doc_type)} for the six Part 2 uploads."""
    exp: dict[str, tuple[str, str | None]] = {
        f"{subject_id}_{suffix}": ("doc", doc_type)
        for doc_type, suffix in DOC_SUFFIXES.items()
    }
    exp[f"{subject_id}_claude_log.jsonl"] = ("log", None)
    return exp


def reset() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.p_stage = "identify"


def _run_hidden_grading(subject_id: str, notebook_text: str) -> None:
    """Grade the notebook against the active rubric. Failures are recorded for the
    admin, never surfaced to the participant."""
    rubric = get_rubric(ACTIVE_RUBRIC_NAME)
    if rubric is None or not (rubric.get("rubric_csv") or "").strip():
        set_participant_grading_error(
            subject_id, f"No usable rubric '{ACTIVE_RUBRIC_NAME}' configured."
        )
        return
    try:
        results = grading.grade_notebook(
            get_client(), rubric["task"], rubric["rubric_csv"], notebook_text
        )
        total = round(sum(r["score"] for r in results), 2)
        mx = round(sum(r["max_pts"] for r in results), 2)
        ok, err = update_participant_grading(
            subject_id, ACTIVE_RUBRIC_NAME, grading.MODEL, results, total, mx
        )
        if not ok:
            set_participant_grading_error(subject_id, err or "grading save failed")
    except Exception as e:
        set_participant_grading_error(subject_id, f"{type(e).__name__}: {e}")


if "p_stage" not in st.session_state:
    st.session_state.p_stage = "identify"

st.title("📥 Study Submission")

# ---- Stage: identify ------------------------------------------------------
if st.session_state.p_stage == "identify":
    subject_id = st.text_input("Subject ID", placeholder="P001").strip()
    id_ok = bool(SUBJECT_ID_RE.match(subject_id))
    if subject_id and not id_ok:
        st.error("Subject ID must be the letter P followed by 3 digits (e.g. P001).")

    existing = get_participant(subject_id) if id_ok else None
    status = existing["status"] if existing else None

    # --- brand-new participant: Part 1 ---
    if id_ok and existing is None:
        st.markdown("**New submission — Part 1.** Upload your notebook and take a short quiz.")
        condition_text = st.text_input(
            "Condition (as assigned to you)", placeholder="e.g. control"
        ).strip()
        condition = normalize_condition(condition_text)
        if condition_text and condition is None:
            st.error("Unrecognized condition. Enter the exact condition you were assigned.")
        if st.button("Start Part 1", type="primary", disabled=not condition):
            ok, err = upsert_participant(subject_id, condition)
            if not ok:
                st.error(f"Could not start the submission: {err}")
                st.stop()
            st.session_state.subject_id = subject_id
            st.session_state.condition = condition
            st.session_state.p_stage = "nb_upload"
            st.rerun()

    # --- started Part 1 but didn't finish the quiz ---
    elif status == "in_progress":
        st.info("You started Part 1 but didn't finish the quiz. You'll restart Part 1.")
        if st.button("Resume Part 1", type="primary"):
            st.session_state.subject_id = subject_id
            st.session_state.condition = existing["condition"]
            st.session_state.p_stage = "nb_upload"
            st.rerun()

    # --- Part 1 done, needs Part 2 ---
    elif status == "quiz_done":
        st.success(
            f"Part 1 is complete for **{subject_id}**. Upload your reflection files "
            "and session log to finish."
        )
        if st.button("Continue to Part 2", type="primary"):
            st.session_state.subject_id = subject_id
            st.session_state.condition = existing["condition"]
            st.session_state.p_stage = "docs_upload"
            st.rerun()

    # --- already fully submitted ---
    elif status == "complete":
        st.warning(
            f"A complete submission for **{subject_id}** already exists. Continuing "
            "replaces it entirely (both parts)."
        )
        if st.checkbox("Replace the existing submission"):
            if st.button("Start over", type="primary"):
                ok, err = upsert_participant(subject_id, existing["condition"])
                if not ok:
                    st.error(f"Could not reset the submission: {err}")
                    st.stop()
                st.session_state.subject_id = subject_id
                st.session_state.condition = existing["condition"]
                st.session_state.p_stage = "nb_upload"
                st.rerun()

# ---- Stage: notebook upload (Part 1) ------------------------------------
elif st.session_state.p_stage == "nb_upload":
    subject_id = st.session_state.subject_id
    nb_name = notebook_filename(subject_id)
    st.markdown(
        f"**Part 1 · Subject {subject_id}.** Upload your notebook — it must be named "
        f"`{nb_name}` exactly."
    )
    uploaded = st.file_uploader("Notebook", type=["ipynb"], accept_multiple_files=True)
    names = {f.name for f in (uploaded or [])}

    ok_names = names == {nb_name}
    for name in sorted(names):
        st.markdown(("✅ " if name == nb_name else "🚫 ") + f"`{name}`"
                    + ("" if name == nb_name else " — remove or rename this file"))
    if not names:
        st.info(f"Select `{nb_name}`.")
    elif not ok_names and nb_name not in names:
        st.info(f"Expected exactly `{nb_name}`.")

    if st.button("Submit notebook", type="primary", disabled=not ok_names):
        try:
            notebook_text = notebook_to_text(next(f for f in uploaded if f.name == nb_name).getvalue())
        except Exception as e:
            st.error(f"Could not read {nb_name}: {e}")
            st.stop()
        if not notebook_text.strip():
            st.error(f"{nb_name} appears to be empty.")
            st.stop()
        ok, err = save_participant_notebook(subject_id, nb_name, notebook_text)
        if not ok:
            st.error(f"Could not save the notebook: {err}")
            st.stop()
        st.session_state.notebook_text = notebook_text
        st.session_state.notebook_filename = nb_name
        st.session_state.p_stage = "quiz"
        st.rerun()

# ---- Stage: quiz (Part 1) ---------------------------------------------
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

# ---- Stage: finalize Part 1 (hidden grading) -------------------------
elif st.session_state.p_stage == "finalize":
    subject_id = st.session_state.subject_id
    with st.spinner("Finalizing Part 1…"):
        _run_hidden_grading(subject_id, st.session_state.notebook_text)
        mark_participant_stage1_done(subject_id)
    st.session_state.p_stage = "stage1_done"
    st.rerun()

# ---- Stage: Part 1 done ---------------------------------------------
elif st.session_state.p_stage == "stage1_done":
    st.success(
        f"✅ Part 1 received for **{st.session_state.subject_id}** — notebook and quiz "
        "complete."
    )
    st.markdown(
        "**Part 2:** upload your five reflection documents and your Claude Code "
        "session log. You can do that now, or come back to this page any time and "
        "enter the same Subject ID."
    )
    col_a, col_b = st.columns(2)
    if col_a.button("Continue to Part 2 now", type="primary"):
        st.session_state.p_stage = "docs_upload"
        st.rerun()
    if col_b.button("I'll come back later"):
        reset()
        st.rerun()

# ---- Stage: documents + log upload (Part 2) --------------------------
elif st.session_state.p_stage == "docs_upload":
    subject_id = st.session_state.subject_id
    exp = stage2_expected(subject_id)
    st.markdown(
        f"**Part 2 · Subject {subject_id}.** Select all **6 files** — each must be "
        f"named `{subject_id}_<name>` exactly:"
    )
    st.code("\n".join(sorted(exp)), language="text")

    uploaded = st.file_uploader(
        "Reflection files + session log", type=["md", "jsonl"], accept_multiple_files=True
    )
    by_name = {f.name: f for f in (uploaded or [])}
    missing = [n for n in exp if n not in by_name]
    unexpected = [n for n in by_name if n not in exp]

    st.markdown("#### File check")
    for name in sorted(exp):
        st.markdown(("✅ " if name in by_name else "❌ ") + f"`{name}`")
    for name in sorted(unexpected):
        st.markdown(f"🚫 unexpected: `{name}` — remove or rename this file")

    ready = not missing and not unexpected
    if not ready:
        st.info("Submit unlocks once exactly the 6 correctly-named files are selected.")

    if st.button("Submit files", type="primary", disabled=not ready):
        errors: list[str] = []
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

            log_name = f"{subject_id}_claude_log.jsonl"
            raw_jsonl = by_name[log_name].getvalue().decode("utf-8", errors="replace")
            metrics = compute_log_metrics(parse_jsonl(raw_jsonl))
            ok, err = save_participant_log(subject_id, log_name, raw_jsonl, metrics)
            if not ok:
                errors.append(f"{log_name}: {err}")

            manifest = {
                "notebook": notebook_filename(subject_id),
                "part2": sorted(exp),
                "received": sorted(by_name),
            }
            mark_participant_complete(subject_id, manifest)

        if errors:
            st.error("Some files could not be saved:\n\n" + "\n".join(errors))
            st.stop()
        st.session_state.p_stage = "complete"
        st.rerun()

# ---- Stage: complete ---------------------------------------------
elif st.session_state.p_stage == "complete":
    st.success(
        f"✅ Submission complete for **{st.session_state.subject_id}** — thank you!"
    )
    st.caption("You can close this tab. Nothing further is required.")
    st.button("Start another submission", on_click=reset)
