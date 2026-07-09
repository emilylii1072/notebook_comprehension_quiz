"""Notebook Quiz — a comprehension quiz generated from an uploaded Jupyter notebook.

A candidate uploads the attrition-model notebook they submitted; the model generates
a fixed 10-question quiz grounded in that specific notebook (two of the questions are
dynamic follow-ups generated from the candidate's own live answer); the candidate
answers one question at a time while a stopwatch (not a countdown — there is no time
limit) tracks elapsed time; correct answers are never revealed until the final
question breakdown.

Run with:  streamlit run app.py
Auth:      set OPENAI_API_KEY (e.g. in a local .env file).
"""

import json
import time

import nbformat
import openai
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()

MODEL = "gpt-4o"  # swap to "gpt-4o-mini" for cheaper/faster test iteration
MAX_OUTPUT_CHARS_PER_CELL = 1500

TASK_CONTEXT = """\
The notebook author was given this task:

"You have been tasked with a PM request to design and create an MVP employee \
attrition model using Viva Insights data. The goal is to build a prototype that \
will inform whether this feature is worth investing engineering resources.
The PM is interested in understanding:
1. If we were to create an employee attrition model, what would this look like?
2. What would be the inputs and outputs of the model?
3. What factors should we take into account when thinking about bringing this \
model into production?
A starter sample Person Query dataset (pq_data) is available via the vivainsights \
Python package. Note: this dataset is known to be incomplete. An attrition label \
column is missing and will need to be engineered or simulated — identifying and \
addressing this gap is part of the task.
The notebook should include: the recommended choice of attrition model algorithm \
and rationale; code demonstrating how the model would run end-to-end; example \
outputs (predictions, feature importance, evaluation metrics); and a brief \
discussion of production considerations."
"""

# Fixed 10-slot quiz structure. Slots in SLOT_ORDER are generated together in one
# batch call (generate_quiz). "model_followup" and "metric_followup" are generated
# on the fly, immediately after the candidate answers "model_choice" / "metric_choice"
# respectively, so the follow-up can reference their own answer.
SLOT_ORDER = [
    "eda_code",
    "missing_labels",
    "model_choice",
    "training_code",
    "metric_choice",
    "outline",
    "io",
    "production",
]

# Where the two dynamic follow-ups are inserted relative to their trigger slot.
FOLLOWUP_AFTER = {
    "model_choice": "model_followup",
    "metric_choice": "metric_followup",
}

SLOT_INSTRUCTIONS = {
    "eda_code": """\
1. eda_code — Find ONE real line of code the notebook actually used during \
exploratory data analysis (e.g., checking missingness, distributions, correlations, \
value counts). Quote it VERBATIM, copied character-for-character from the notebook \
transcript above, wrapped in single backticks, inside the question text, framed with \
what it was checking (e.g., "Which line of code did you use to check for missing \
values in the dataset?"). The 4 options are candidate lines of code, each a short \
single line wrapped in single backticks: one is the exact real line copied verbatim \
from the notebook; the other three are similar-looking but different lines (different \
method, column, or argument) that would NOT actually appear in this notebook. \
correct_index points to the real line.""",
    "missing_labels": """\
2. missing_labels — Ask how the notebook addressed the missing attrition label. \
Ground the 4 options in what THIS notebook actually did versus plausible alternatives \
it did not do (e.g., dropping unlabeled rows, using an unrelated column as the label, \
simulating a label from proxy signals, manual labeling).""",
    "model_choice": """\
3. model_choice — Ask which algorithm they used/chose as their (primary or \
recommended) attrition model. Exactly one option must be an algorithm the notebook \
actually trained; the other three must be plausible ML algorithms that do NOT appear \
anywhere in the notebook.""",
    "training_code": """\
4. training_code — Quote a real, short code snippet (2-6 lines, copied VERBATIM \
character-for-character from the notebook transcript above, no paraphrasing) from \
the notebook's model-training section, formatted as a fenced ```python code block \
inside the question text (e.g., "What does the following code do?"). The 4 options \
are candidate descriptions of what that snippet does: one correct, three \
plausible-but-wrong (e.g., confusing training with evaluation, tuning, or \
saving/loading a model).""",
    "metric_choice": """\
5. metric_choice — Ask which evaluation metric(s) they used to assess the model. \
Exactly one option must be a metric the notebook actually reported; the other three \
must be plausible metrics the notebook did NOT compute.""",
    "outline": """\
6. outline — Ask the reader to identify the actual structure of their notebook. The \
4 options are each a short outline of the notebook's sections written as ONE single \
line using " → " between sections (e.g., "1. Load data → 2. EDA → 3. Label \
simulation → 4. Model training → 5. Evaluation → 6. Production considerations") — do \
NOT use newlines inside an option. Exactly one outline must match the real \
order/sections of the notebook above; the other three must be plausible-but-wrong \
orderings or section sets (reordered steps, an invented section, or a real section \
removed/renamed).""",
    "io": """\
7. io — Ask what the final model's inputs and/or outputs actually are (e.g., which \
features feed it, what it predicts/returns). Ground the correct option in what the \
notebook actually implemented; distractors name inputs/outputs the notebook does not \
produce.""",
    "production": """\
8. production — Ask what the notebook said about considerations for bringing this \
model into production (e.g., label quality, bias/fairness, monitoring/drift, privacy, \
threshold setting). Ground the correct option in what the notebook actually \
discussed; distractors are plausible production concerns the notebook did NOT \
mention.""",
}

SHARED_REQUIREMENTS = """\
Shared requirements for every question:
- Exactly 4 options per question, exactly one correct (correct_index is 0-based).
- Ground every question in the specific content of THIS notebook: reference its \
actual modeling choices, engineered attrition label, feature names, metric values, \
and stated production considerations.
- Each question must be answerable in about 45-60 seconds without re-running code.
- Do NOT ask trivia (import order, variable names, library versions).
- topic: a 2-4 word tag. explanation: 1-3 sentences on why the correct answer is \
right, shown to the candidate in the question breakdown after they finish.
- Distractors must be plausible — they mix up related concepts from the same \
notebook — but clearly wrong to someone who understands the work.
"""

SYSTEM_PROMPT = (
    "You are an expert technical interviewer for data science roles. You evaluate "
    "whether a candidate genuinely understands a take-home notebook they submitted "
    "— including work they may have produced with AI assistance. You ask precise, "
    "notebook-specific questions and give calibrated, evidence-based assessments.\n\n"
    + TASK_CONTEXT
)

FOLLOWUP_SUBJECT_LABEL = {
    "model": "algorithm/model",
    "metric": "evaluation metric",
}


class QuizQuestion(BaseModel):
    slot: str
    topic: str
    question: str
    options: list[str]
    correct_index: int
    explanation: str


class Quiz(BaseModel):
    questions: list[QuizQuestion]


class SingleQuestion(BaseModel):
    question: QuizQuestion


# ---------------------------------------------------------------------------
# Notebook parsing
# ---------------------------------------------------------------------------

def notebook_to_text(raw: bytes) -> str:
    """Flatten a .ipynb into a text transcript of markdown, code, and outputs."""
    nb = nbformat.reads(raw.decode("utf-8"), as_version=4)
    parts = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown":
            parts.append(f"--- markdown cell {i} ---\n{cell.source}")
        elif cell.cell_type == "code":
            parts.append(f"--- code cell {i} ---\n{cell.source}")
            out_texts = []
            for out in cell.get("outputs", []):
                text = ""
                if out.get("output_type") == "stream":
                    text = "".join(out.get("text", ""))
                elif out.get("output_type") in ("execute_result", "display_data"):
                    text = "".join(out.get("data", {}).get("text/plain", ""))
                elif out.get("output_type") == "error":
                    text = "\n".join(out.get("traceback", []))
                if text:
                    if len(text) > MAX_OUTPUT_CHARS_PER_CELL:
                        text = text[:MAX_OUTPUT_CHARS_PER_CELL] + "\n[output truncated]"
                    out_texts.append(text)
            if out_texts:
                parts.append(f"--- output of cell {i} ---\n" + "\n".join(out_texts))
    return "\n\n".join(parts)


def notebook_prompt_prefix(notebook_text: str) -> str:
    """The notebook wrapped for the prompt, shared verbatim by every API call.

    Every call uses an identical system prompt and leads with this exact text, so
    OpenAI's automatic prefix caching (prefixes over ~1024 tokens, unchanged across
    requests) can apply to the follow-up and grading calls.
    """
    return f"<notebook>\n{notebook_text}\n</notebook>"


BATCH_JSON_SCHEMA_NOTE = """\
Respond with ONLY a single JSON object (no markdown code fences, no commentary) \
matching this exact shape:
{"questions": [
  {"slot": "<one of: eda_code, missing_labels, model_choice, training_code, \
metric_choice, outline, io, production>",
   "topic": "<2-4 word tag>", "question": "<question text, may include \\n and \
fenced ```code``` blocks>",
   "options": ["<A>", "<B>", "<C>", "<D>"], "correct_index": <0-3 integer>,
   "explanation": "<1-3 sentence explanation>"}
]}
Return EXACTLY 8 questions, one per slot listed above, in that exact order \
(eda_code first, production last).
"""


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------

def generate_quiz(client: OpenAI, notebook_text: str) -> list[QuizQuestion]:
    """Generate the 8 static (non-follow-up) questions in one call."""
    instructions = "\n\n".join(SLOT_INSTRUCTIONS[s] for s in SLOT_ORDER)
    instructions = f"{instructions}\n\n{SHARED_REQUIREMENTS}\n{BATCH_JSON_SCHEMA_NOTE}"
    user_content = f"{notebook_prompt_prefix(notebook_text)}\n\n{instructions}"

    response = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    choice = response.choices[0]
    if choice.finish_reason == "content_filter":
        raise RuntimeError("The model declined to process this notebook (content filter).")

    try:
        quiz = Quiz.model_validate(json.loads(choice.message.content))
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        raise RuntimeError(f"Could not parse quiz JSON from the model: {e}") from e

    order_index = {slot: i for i, slot in enumerate(SLOT_ORDER)}
    seen = set()
    deduped = []
    for q in quiz.questions:
        if q.slot in order_index and q.slot not in seen:
            if len(q.options) != 4 or not (0 <= q.correct_index < 4):
                raise RuntimeError(
                    f"The generated '{q.slot}' question was malformed — try again."
                )
            seen.add(q.slot)
            deduped.append(q)
    deduped.sort(key=lambda q: order_index[q.slot])

    if len(deduped) != len(SLOT_ORDER):
        missing = [s for s in SLOT_ORDER if s not in seen]
        raise RuntimeError(
            f"Quiz generation was missing required sections: {', '.join(missing)} — "
            "try again."
        )
    return deduped


def generate_followup(
    client: OpenAI, notebook_text: str, subject: str, chosen_label: str | None
) -> QuizQuestion:
    """Generate the model/metric follow-up question, based on the candidate's answer
    to the preceding model_choice/metric_choice question (or a notebook-grounded
    fallback if they left it unanswered)."""
    label = FOLLOWUP_SUBJECT_LABEL[subject]
    if chosen_label:
        framing = (
            f'The candidate just selected "{chosen_label}" as the {label} they used '
            f'for this task. Write ONE follow-up multiple-choice question asking why '
            f'"{chosen_label}" would be considered ideal/appropriate for this '
            "attrition-prediction task. Exactly one option must be a TRUE statement "
            f'about "{chosen_label}" relevant to this kind of task; the other three '
            f"must be plausible-sounding but FALSE statements (e.g., true of a "
            f'different {label}, or simply incorrect). If the notebook explicitly '
            f'states a rationale for choosing "{chosen_label}", prefer that rationale '
            "as the correct option."
        )
    else:
        framing = (
            "The candidate left the previous question unanswered. Instead, write ONE "
            f"follow-up multiple-choice question that tests whether the reader "
            f"understands why the {label} actually used in the notebook above was "
            "appropriate for this task. Ground the correct option in the notebook's "
            f"own reasoning where stated, otherwise in general ML knowledge about "
            f"that {label}."
        )
    instructions = f"""\
{framing}

{SHARED_REQUIREMENTS}
Respond with ONLY a single JSON object (no markdown code fences, no commentary) \
matching this exact shape:
{{"question": {{"slot": "{subject}_followup", "topic": "<2-4 word tag>", \
"question": "<question text>", "options": ["<A>", "<B>", "<C>", "<D>"], \
"correct_index": <0-3 integer>, "explanation": "<1-3 sentence explanation>"}}}}
"""
    user_content = f"{notebook_prompt_prefix(notebook_text)}\n\n{instructions}"

    response = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    choice = response.choices[0]
    if choice.finish_reason == "content_filter":
        raise RuntimeError(
            "The model declined to generate a follow-up question (content filter)."
        )
    try:
        wrapped = SingleQuestion.model_validate(json.loads(choice.message.content))
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        raise RuntimeError(f"Could not parse follow-up question JSON: {e}") from e

    q = wrapped.question
    if len(q.options) != 4 or not (0 <= q.correct_index < 4):
        raise RuntimeError("The generated follow-up question was malformed — try again.")
    return q


def call_with_errors_surfaced(fn, *args, **kwargs):
    """Run an API-calling function, converting SDK errors to readable messages."""
    try:
        return fn(*args, **kwargs)
    except openai.AuthenticationError:
        st.error(
            "Authentication failed. Set the OPENAI_API_KEY environment variable "
            "(e.g. in a local .env file) and restart the app."
        )
    except openai.RateLimitError:
        st.error("Rate limited (or out of quota) — check your OpenAI plan/billing and try again.")
    except openai.APIStatusError as e:
        st.error(f"API error {e.status_code}: {e.message}")
    except openai.APIConnectionError:
        st.error("Could not reach the OpenAI API — check your network connection.")
    except RuntimeError as e:
        st.error(str(e))
    return None


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Notebook Comprehension Quiz", page_icon="📝", layout="centered")

if "stage" not in st.session_state:
    st.session_state.stage = "upload"


def reset():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.stage = "upload"


@st.cache_resource
def get_client() -> OpenAI:
    return OpenAI()


def build_items(static_questions: list[QuizQuestion]) -> list:
    """Interleave the 8 static questions with the two follow-up placeholders,
    producing the fixed 10-slot sequence in template order."""
    items = []
    for q in static_questions:
        items.append(q)
        if q.slot in FOLLOWUP_AFTER:
            items.append({"placeholder": True, "subject": q.slot.replace("_choice", "")})
    return items


def record_current_answer():
    """Save the radio selection (and elapsed time) for the current question index."""
    idx = st.session_state.current_index
    item = st.session_state.quiz_items[idx]
    # Streamlit re-executes the whole script (including class definitions) on every
    # rerun, so a QuizQuestion instance stored in session_state from an earlier rerun
    # is never `isinstance` of *this* rerun's freshly-defined QuizQuestion class.
    # Check against the stable builtin `dict` (the placeholder marker) instead.
    if not isinstance(item, dict):
        selected = st.session_state.get(f"q{idx}")
        st.session_state.answers[idx] = (
            item.options.index(selected) if selected in item.options else None
        )
        started = st.session_state.question_start_times.get(idx, time.time())
        st.session_state.time_spent[idx] = round(time.time() - started, 1)


def finish_quiz():
    """Collect only materialized questions (drops any never-reached follow-up
    placeholder, and any question explicitly marked as a failed/skipped generation)."""
    final_q, final_a, final_t = [], [], []
    for it, a, t in zip(
        st.session_state.quiz_items, st.session_state.answers, st.session_state.time_spent
    ):
        if not isinstance(it, dict) and it.topic != "skipped":
            final_q.append(it)
            final_a.append(a)
            final_t.append(t)
    st.session_state.final_questions = final_q
    st.session_state.final_answers = final_a
    st.session_state.final_time_spent = final_t
    st.session_state.elapsed = time.time() - st.session_state.started_at
    st.session_state.stage = "results"


st.title("📝 Notebook Comprehension Quiz")

# ---- Stage: upload -------------------------------------------------------
if st.session_state.stage == "upload":
    st.markdown(
        "Upload the Jupyter notebook you submitted for the **employee attrition "
        "model** task. A 10-question quiz (two of which are generated live from "
        "your own answers) will check your understanding of your own submission. "
        "A stopwatch tracks how long you take, but there's no time limit."
    )

    uploaded = st.file_uploader("Notebook (.ipynb)", type=["ipynb"])

    if uploaded is not None and st.button("Generate quiz", type="primary"):
        try:
            notebook_text = notebook_to_text(uploaded.getvalue())
        except Exception as e:
            st.error(f"Could not parse the notebook: {e}")
            st.stop()
        if not notebook_text.strip():
            st.error("The notebook appears to be empty.")
            st.stop()

        with st.spinner("Reading the notebook and generating questions… (~1 minute)"):
            static_questions = call_with_errors_surfaced(
                generate_quiz, get_client(), notebook_text
            )
        if static_questions is not None:
            items = build_items(static_questions)
            st.session_state.quiz_items = items
            st.session_state.answers = [None] * len(items)
            st.session_state.time_spent = [None] * len(items)
            st.session_state.question_start_times = {}
            st.session_state.notebook_text = notebook_text
            st.session_state.current_index = 0
            st.session_state.started_at = time.time()
            st.session_state.stage = "quiz"
            st.rerun()

# ---- Stage: quiz ---------------------------------------------------------
elif st.session_state.stage == "quiz":
    idx = st.session_state.current_index

    item = st.session_state.quiz_items[idx]

    # Materialize a follow-up placeholder based on the candidate's answer to the
    # preceding trigger question, the first time we land on this index.
    if isinstance(item, dict) and item.get("placeholder"):
        subject = item["subject"]
        trigger_item = st.session_state.quiz_items[idx - 1]
        trigger_answer_idx = st.session_state.answers[idx - 1]
        chosen_label = (
            trigger_item.options[trigger_answer_idx]
            if not isinstance(trigger_item, dict) and trigger_answer_idx is not None
            else None
        )
        with st.spinner("Generating a follow-up based on your answer…"):
            followup = call_with_errors_surfaced(
                generate_followup, get_client(), st.session_state.notebook_text,
                subject, chosen_label,
            )
        if followup is None:
            col_a, col_b = st.columns(2)
            if col_a.button("Retry"):
                st.rerun()
            if col_b.button("Skip this question"):
                st.session_state.quiz_items[idx] = QuizQuestion(
                    slot=f"{subject}_followup",
                    topic="skipped",
                    question="_(This follow-up could not be generated and was skipped.)_",
                    options=["N/A", "N/A", "N/A", "N/A"],
                    correct_index=0,
                    explanation="Generation failed; this question was skipped.",
                )
                st.session_state.current_index += 1
                st.rerun()
            st.stop()
        st.session_state.quiz_items[idx] = followup
        item = followup

    if idx not in st.session_state.question_start_times:
        st.session_state.question_start_times[idx] = time.time()

    @st.fragment(run_every=1.0)
    def stopwatch():
        elapsed = time.time() - st.session_state.started_at
        mm, ss = divmod(int(elapsed), 60)
        st.metric("⏱️ Time elapsed", f"{mm}:{ss:02d}")

    stopwatch()
    st.divider()

    st.subheader(f"Question {idx + 1} of {len(st.session_state.quiz_items)}")
    st.markdown(item.question)
    st.radio(
        "Select one:",
        item.options,
        index=None,
        key=f"q{idx}",
        label_visibility="collapsed",
    )

    is_last = idx == len(st.session_state.quiz_items) - 1
    if st.button("Submit answers" if is_last else "Next question", type="primary"):
        record_current_answer()
        if is_last:
            finish_quiz()
        else:
            st.session_state.current_index += 1
        st.rerun()

# ---- Stage: results ------------------------------------------------------
elif st.session_state.stage == "results":
    questions = st.session_state.final_questions
    answers = st.session_state.final_answers
    time_spent = st.session_state.final_time_spent
    score = sum(1 for q, a in zip(questions, answers) if a == q.correct_index)
    total = len(questions)

    col1, col2 = st.columns(2)
    col1.metric("Score", f"{score} / {total}")
    col2.metric("Time used", f"{st.session_state.elapsed:.0f}s")

    st.markdown("## Question breakdown")
    for i, (q, a, t) in enumerate(zip(questions, answers, time_spent)):
        correct = a == q.correct_index
        icon = "✅" if correct else ("⬜" if a is None else "❌")
        with st.expander(f"{icon} Q{i + 1} · {q.topic}"):
            st.markdown(q.question)
            st.markdown(f"- **Your answer:** {q.options[a] if a is not None else '_unanswered_'}")
            st.markdown(f"- **Correct answer:** {q.options[q.correct_index]}")
            st.markdown(f"- **Why:** {q.explanation}")
            st.markdown(f"- **Time spent:** {t:.0f}s" if t is not None else "- **Time spent:** _unknown_")

    results_payload = {
        "score": score,
        "total": total,
        "elapsed_seconds": round(st.session_state.elapsed, 1),
        "questions": [
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
            for q, a, t in zip(questions, answers, time_spent)
        ],
    }
    st.download_button(
        "Download results (JSON)",
        data=json.dumps(results_payload, indent=2),
        file_name="quiz_results.json",
        mime="application/json",
    )
    st.button("Start over", on_click=reset)
