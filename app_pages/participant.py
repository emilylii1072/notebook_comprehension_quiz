"""Participant intake — the only participant-facing page.

One continuous run, keyed to the subject ID entered at the start. The stage the
participant is on is written to `participants.stage` as they go, so re-entering
the same ID resumes exactly where they left off rather than at a coarse
checkpoint:

    identify        subject ID + assigned condition
    pre_survey      lib/surveys.py PRE_SURVEY_ITEMS
    main_task       the condition's task document, then a 40-minute countdown
    main_upload     plan + notebook (exact names) + any extra files
    ideate_task     the ideation document, untimed but measured. Nothing is
                    uploaded: the participant pitches their idea aloud and the
                    admin uploads the transcript afterwards (Admin > Participant
                    > Ideate), where it is graded.
    quiz           the notebook comprehension MCQ
    interview       hand-off screen; the interview happens away from the app
    debug_task      the debugging document, then a 15-minute countdown
    debug_upload    one debugging markdown document
    log_upload      the Claude Code session log
    post_survey     lib/surveys.py POST_SURVEY_ITEMS
    complete        hidden grading runs here, then done

Task instructions are admin-authored (Admin > Task instructions). Timers are
advisory: the countdown beeps and says time is up, but nothing locks -- see
lib/tasks.render_countdown. Timing is anchored to a stored started_at, so a
refresh doesn't hand anyone a fresh allowance.

No score, breakdown, or grading error is ever shown to the participant — all of
that lives behind the admin gate.

One page of the multipage app — run via `streamlit run app.py`.
Auth: set ANTHROPIC_API_KEY (quiz generation + hidden grading).
"""

import re

import streamlit as st
from dotenv import load_dotenv

from db import (
    finish_participant_task,
    get_participant,
    get_participant_survey,
    get_participant_task_timing,
    get_rubric,
    get_secret,
    get_task_instruction,
    mark_participant_complete,
    save_participant_extra_file,
    save_participant_file,
    save_participant_log,
    save_participant_notebook,
    set_participant_grading_error,
    set_participant_stage,
    start_participant_task,
    update_participant_grading,
    upsert_participant,
)
from lib import grading, tasks
from lib.llm import get_client
from lib.notebook import notebook_to_text
from lib.quiz_ui import render_quiz_flow
from lib.surveys import POST_SURVEY_ITEMS, PRE_SURVEY_ITEMS
from lib.survey_ui import render_survey_flow
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

# The run, in order. `status` is what the Admin overview filters on; it moves
# from in_progress to quiz_done at the MCQ and to complete at the end, matching
# what those values have always meant.
STAGES = [
    "pre_survey", "main_task", "main_upload", "ideate_task",
    "quiz", "interview", "debug_task", "debug_upload", "log_upload",
    "post_survey", "complete",
]
STAGE_STATUS = {"complete": "complete"}
# Stages that no longer exist, for participants whose stored `stage` predates the
# change. The ideation upload screen is gone, so they carry on from the quiz.
RETIRED_STAGES = {"ideate_upload": "quiz"}

# Progress labels — the participant sees how far along they are, never a score.
STAGE_STEP = {
    "pre_survey": ("Opening survey", 1),
    "main_task": ("Main task", 2), "main_upload": ("Main task", 2),
    "ideate_task": ("Idea generation", 3),
    "quiz": ("Questions about your notebook", 4),
    "interview": ("Interview", 5),
    "debug_task": ("Debugging", 6), "debug_upload": ("Debugging", 6),
    "log_upload": ("Session log", 7),
    "post_survey": ("Closing survey", 8),
}
N_STEPS = 8

# doc_type -> filename suffix, for the documents a participant uploads. The
# manual/AI split (debug_manual / debug_ai / ideate_manual / ideate_ai) belonged to
# an earlier protocol and is no longer collected, and neither is an ideation
# write-up (the pitch is transcribed by the admin), though Admin still shows them
# for participants who were run under those protocols.
DOC_SUFFIXES = {
    "task_plan": "task_plan.md",
    "debug": "debug.md",
}
DOC_TITLES = {
    "task_plan": "Task plan",
    "debug": "Debugging write-up",
}


def normalize_condition(text: str) -> str | None:
    key = re.sub(r"[\s_-]+", " ", (text or "").strip().lower()).strip()
    return CONDITION_ALIASES.get(key) or CONDITION_ALIASES.get(key.replace(" ", ""))


def notebook_filename(subject_id: str) -> str:
    return f"{subject_id}_notebook.ipynb"


def doc_filename(subject_id: str, doc_type: str) -> str:
    return f"{subject_id}_{DOC_SUFFIXES[doc_type]}"


def log_filename(subject_id: str) -> str:
    return f"{subject_id}_claude_log.jsonl"


def reset() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.p_stage = "identify"


def goto(stage: str) -> None:
    """Advance to `stage`, remembering it so the participant can resume there."""
    sid = st.session_state.get("subject_id")
    if sid:
        set_participant_stage(sid, stage, STAGE_STATUS.get(stage))
    st.session_state.p_stage = stage
    st.rerun()


def _decode(upload) -> str:
    raw = upload.getvalue()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


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
            get_client(), rubric["task"], rubric["rubric_csv"], notebook_text,
            instructions=rubric.get("grading_instructions"),
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


def render_task_screen(*, task_key: str, next_stage: str) -> None:
    """Instructions, then a Start button, then the clock. Shared by all three
    tasks; the only differences are which document is shown and whether there's
    an allowance to count down."""
    sid = st.session_state.subject_id
    condition = st.session_state.condition
    limit = tasks.TASK_LIMIT_SECONDS[task_key]

    instructions = [
        get_task_instruction(k) for k in tasks.instruction_keys_for(task_key, condition)
    ]
    readable = tasks.render_instructions(instructions, task_key)
    if task_key == tasks.DEBUG:
        tasks.render_notebook_link(get_task_instruction(tasks.DEBUG_NOTEBOOK_KEY))

    # Session state is empty in a browser session that didn't start the task, so
    # fall back to the stored row: someone resuming a task they already began
    # must get their running clock back, not a fresh allowance.
    timing = st.session_state.get(f"timing_{task_key}")
    if timing is None:
        timing = get_participant_task_timing(sid, task_key)
        if timing is not None:
            st.session_state[f"timing_{task_key}"] = timing
    if timing is None:
        st.divider()
        if limit is None:
            st.markdown(
                "**There's no time limit on this task** — we only record how long "
                "it takes. Start when you're ready to begin."
            )
        else:
            st.markdown(
                f"**You'll have {tasks.format_duration(limit)}.** The timer starts "
                "when you click below and will beep when the time is up — you can "
                "still finish and upload after it does."
            )
        if st.button("Start the task", type="primary", disabled=not readable):
            st.session_state[f"timing_{task_key}"] = start_participant_task(
                sid, task_key, limit
            )
            st.rerun()
        return

    started = _started_epoch(timing)
    tasks.render_countdown(started_at_epoch=started, limit_seconds=limit)
    st.caption(
        "Leave this page open while you work. If it closes, re-enter your "
        "Subject ID — the clock keeps running from when you started."
    )
    if st.button("I've finished — continue", type="primary"):
        finish_participant_task(sid, task_key)
        goto(next_stage)


def _started_epoch(timing: dict) -> float:
    """started_at as a POSIX timestamp. Supabase returns ISO-8601 with a Z or a
    +00:00 offset depending on the column; both parse once Z is normalised."""
    from datetime import datetime, timezone

    raw = (timing or {}).get("started_at")
    if not raw:
        return datetime.now(timezone.utc).timestamp()
    try:
        text = str(raw).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return datetime.now(timezone.utc).timestamp()


def render_single_doc_upload(*, doc_type: str, task_key: str, next_stage: str) -> None:
    """Upload exactly one markdown document, named for the participant."""
    sid = st.session_state.subject_id
    want = doc_filename(sid, doc_type)
    st.markdown(
        f"Upload your **{DOC_TITLES[doc_type].lower()}** — it must be named "
        f"`{want}` exactly."
    )
    uploaded = st.file_uploader(DOC_TITLES[doc_type], type=["md"], key=f"up_{doc_type}")
    ok_name = uploaded is not None and uploaded.name == want
    if uploaded is not None and not ok_name:
        st.error(f"🚫 `{uploaded.name}` — rename it to `{want}` and select it again.")
    elif ok_name:
        st.success(f"✅ `{want}`")

    if st.button("Submit and continue", type="primary", disabled=not ok_name):
        ok, err = save_participant_file(sid, doc_type, want, _decode(uploaded))
        if not ok:
            st.error(f"Could not save {want}: {err}")
            st.stop()
        finish_participant_task(sid, task_key)
        goto(next_stage)


if "p_stage" not in st.session_state:
    st.session_state.p_stage = "identify"

st.title("📥 Study Submission")

_stage = st.session_state.p_stage
if _stage in STAGE_STEP:
    _label, _n = STAGE_STEP[_stage]
    st.caption(f"Step {_n} of {N_STEPS} · {_label}")
    st.progress((_n - 1) / N_STEPS)

# ---- Stage: identify ------------------------------------------------------
if _stage == "identify":
    subject_id = st.text_input("Subject ID", placeholder="P001").strip()
    id_ok = bool(SUBJECT_ID_RE.match(subject_id))
    if subject_id and not id_ok:
        st.error("Subject ID must be the letter P followed by 3 digits (e.g. P001).")

    existing = get_participant(subject_id) if id_ok else None
    status = existing["status"] if existing else None

    # --- brand-new participant ---
    if id_ok and existing is None:
        st.markdown(
            "**New submission.** You'll answer a short survey, work through three "
            "tasks with a break for some questions, and finish with a closing survey."
        )
        condition_text = st.text_input(
            "Condition (as assigned to you)", placeholder="e.g. control"
        ).strip()
        condition = normalize_condition(condition_text)
        if condition_text and condition is None:
            st.error("Unrecognized condition. Enter the exact condition you were assigned.")
        if st.button("Begin", type="primary", disabled=not condition):
            ok, err = upsert_participant(subject_id, condition)
            if not ok:
                st.error(f"Could not start the submission: {err}")
                st.stop()
            st.session_state.subject_id = subject_id
            st.session_state.condition = condition
            goto("pre_survey")

    # --- already finished ---
    elif status == "complete":
        st.warning(
            f"A complete submission for **{subject_id}** already exists. Continuing "
            "replaces it entirely."
        )
        if st.checkbox("Replace the existing submission"):
            if st.button("Start over", type="primary"):
                ok, err = upsert_participant(subject_id, existing["condition"])
                if not ok:
                    st.error(f"Could not reset the submission: {err}")
                    st.stop()
                st.session_state.subject_id = subject_id
                st.session_state.condition = existing["condition"]
                goto("pre_survey")

    # --- partway through: resume exactly where they stopped ---
    elif id_ok and existing is not None:
        resume = existing.get("stage")
        resume = RETIRED_STAGES.get(resume, resume)
        if resume not in STAGES:
            # Started before per-stage resume existed, or never got past the
            # opening survey: fall back to the survey, skipping it if it's done.
            resume = "main_task" if get_participant_survey(subject_id, "pre") else "pre_survey"
        label = STAGE_STEP.get(resume, ("Closing survey", N_STEPS))[0]
        st.info(f"Welcome back — you'll pick up at **{label}**.")
        if st.button("Resume", type="primary"):
            st.session_state.subject_id = subject_id
            st.session_state.condition = existing["condition"]
            goto(resume)

# ---- Stage: opening survey -----------------------------------------------
elif _stage == "pre_survey":
    st.markdown(
        "**Before you begin.** A short survey — there's no time limit, but we do "
        "track how long each question takes. You won't see a score."
    )
    if render_survey_flow(
        subject_id=st.session_state.subject_id, survey_type="pre", items=PRE_SURVEY_ITEMS,
    ):
        goto("main_task")

# ---- Stage: main task ----------------------------------------------------
elif _stage == "main_task":
    render_task_screen(task_key=tasks.MAIN, next_stage="main_upload")

# ---- Stage: main task upload ---------------------------------------------
elif _stage == "main_upload":
    sid = st.session_state.subject_id
    nb_name = notebook_filename(sid)
    plan_name = doc_filename(sid, "task_plan")
    st.markdown(
        "**Upload your main-task files.** Two are required, named exactly as shown:"
    )
    st.code(f"{plan_name}\n{nb_name}", language="text")

    required = st.file_uploader(
        "Plan and notebook", type=["md", "ipynb"], accept_multiple_files=True,
        key="up_main_required",
    )
    by_name = {f.name: f for f in (required or [])}
    have_plan, have_nb = plan_name in by_name, nb_name in by_name
    for name, present in ((plan_name, have_plan), (nb_name, have_nb)):
        st.markdown(("✅ " if present else "❌ ") + f"`{name}`")
    for name in sorted(n for n in by_name if n not in (plan_name, nb_name)):
        st.markdown(f"🚫 unexpected: `{name}` — rename it, or add it as an extra file below")

    st.markdown("#### Anything else? (optional)")
    st.caption(
        "Scratch scripts, figures, exports — anything else you produced. Any "
        "filename is fine here."
    )
    extras = st.file_uploader(
        "Additional files", accept_multiple_files=True, key="up_main_extra",
    )
    for f in extras or []:
        st.markdown(f"📎 `{f.name}`")

    ready = have_plan and have_nb
    if not ready:
        st.info("Submit unlocks once both required files are selected.")

    if st.button("Submit and continue", type="primary", disabled=not ready):
        try:
            notebook_text = notebook_to_text(by_name[nb_name].getvalue())
        except Exception as e:
            st.error(f"Could not read {nb_name}: {e}")
            st.stop()
        if not notebook_text.strip():
            st.error(f"{nb_name} appears to be empty.")
            st.stop()

        errors: list[str] = []
        with st.spinner("Saving your files…"):
            ok, err = save_participant_notebook(sid, nb_name, notebook_text)
            if not ok:
                errors.append(f"{nb_name}: {err}")
            ok, err = save_participant_file(
                sid, "task_plan", plan_name, _decode(by_name[plan_name])
            )
            if not ok:
                errors.append(f"{plan_name}: {err}")
            for f in extras or []:
                ok, err = save_participant_extra_file(sid, f.name, f.getvalue())
                if not ok:
                    errors.append(f"{f.name}: {err}")
        if errors:
            st.error("Some files could not be saved:\n\n" + "\n".join(errors))
            st.stop()

        st.session_state.notebook_text = notebook_text
        st.session_state.notebook_filename = nb_name
        finish_participant_task(sid, tasks.MAIN)
        goto("ideate_task")

# ---- Stage: ideation task ------------------------------------------------
elif _stage == "ideate_task":
    render_task_screen(task_key=tasks.IDEATE, next_stage="quiz")

# ---- Stage: MCQ about their own notebook ---------------------------------
elif _stage == "quiz":
    sid = st.session_state.subject_id
    if "notebook_text" not in st.session_state:
        # Resumed in a fresh browser session — the notebook is on file, so pull
        # it back rather than asking for it again.
        from db import get_participant_bundle

        nb = get_participant_bundle(sid)["notebook"]
        if not nb:
            st.error(
                "We couldn't find your notebook on file. Please tell the researcher."
            )
            st.stop()
        st.session_state.notebook_text = nb["notebook_text"]
        st.session_state.notebook_filename = nb["filename"]

    st.markdown(
        "Answer each question about **your own notebook**. There is no time limit — "
        "a stopwatch just tracks how long you take. You won't see a score."
    )
    if render_quiz_flow(
        notebook_text=st.session_state.notebook_text,
        notebook_filename=st.session_state.notebook_filename,
        subject_id=sid,
    ):
        set_participant_stage(sid, "interview", "quiz_done")
        st.session_state.p_stage = "interview"
        st.rerun()

# ---- Stage: interview hand-off -------------------------------------------
elif _stage == "interview":
    st.markdown(
        "### Interview\n\n"
        "Now, to get more of your thoughts on the design, you will answer some "
        "questions verbally in an interview style.\n\n"
        "The researcher will take it from here — nothing to upload for this part."
    )
    st.info("When the interview is over, click below to continue to the last task.")
    if st.button("The interview is over — continue", type="primary"):
        goto("debug_task")

# ---- Stage: debugging task -----------------------------------------------
elif _stage == "debug_task":
    render_task_screen(task_key=tasks.DEBUG, next_stage="debug_upload")

elif _stage == "debug_upload":
    render_single_doc_upload(
        doc_type="debug", task_key=tasks.DEBUG, next_stage="log_upload",
    )

# ---- Stage: session log --------------------------------------------------
elif _stage == "log_upload":
    sid = st.session_state.subject_id
    want = log_filename(sid)
    st.markdown(
        f"**Almost there.** Upload your Claude Code session log — named `{want}` "
        "exactly. If you worked across several sessions, select all of them."
    )
    uploaded = st.file_uploader(
        "Session log", type=["jsonl"], accept_multiple_files=True, key="up_log",
    )
    files = list(uploaded or [])
    # Several sessions are fine, but each file still has to be recognisably this
    # participant's: exactly the canonical name, or that name plus a suffix.
    stem = want[: -len(".jsonl")]
    good = [f for f in files if f.name == want or f.name.startswith(f"{stem}_")]
    bad = [f for f in files if f not in good]
    for f in good:
        st.markdown(f"✅ `{f.name}`")
    for f in bad:
        st.markdown(f"🚫 `{f.name}` — expected `{want}` (or `{stem}_2.jsonl` for a second session)")
    if not files:
        st.info(f"Select `{want}`.")

    if st.button("Submit and continue", type="primary", disabled=not good or bool(bad)):
        errors = []
        with st.spinner("Saving your log…"):
            for f in good:
                raw = f.getvalue().decode("utf-8", errors="replace")
                metrics = compute_log_metrics(parse_jsonl(raw))
                ok, err = save_participant_log(sid, f.name, raw, metrics)
                if not ok:
                    errors.append(f"{f.name}: {err}")
        if errors:
            st.error("Could not save:\n\n" + "\n".join(errors))
            st.stop()
        st.session_state.log_names = [f.name for f in good]
        goto("post_survey")

# ---- Stage: closing survey -----------------------------------------------
elif _stage == "post_survey":
    st.markdown(
        "**Last step.** One more short survey about your experience with the "
        "tasks. There's no time limit, but we do track how long each question "
        "takes. You won't see a score."
    )
    if render_survey_flow(
        subject_id=st.session_state.subject_id, survey_type="post", items=POST_SURVEY_ITEMS,
    ):
        sid = st.session_state.subject_id
        with st.spinner("Finishing up…"):
            if "notebook_text" in st.session_state:
                _run_hidden_grading(sid, st.session_state.notebook_text)
            mark_participant_complete(sid, {
                "notebook": notebook_filename(sid),
                "docs": [doc_filename(sid, d) for d in DOC_SUFFIXES],
                "logs": st.session_state.get("log_names", []),
            })
            set_participant_stage(sid, "complete", "complete")
        st.session_state.p_stage = "complete"
        st.rerun()

# ---- Stage: complete -----------------------------------------------------
elif _stage == "complete":
    st.success(
        f"✅ Submission complete for **{st.session_state.subject_id}** — thank you!"
    )
    st.caption("You can close this tab. Nothing further is required.")
    st.button("Start another submission", on_click=reset)
