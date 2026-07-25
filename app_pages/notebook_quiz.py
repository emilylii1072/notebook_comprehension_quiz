"""Notebook Quiz — a comprehension quiz generated from an uploaded Jupyter notebook.

A candidate uploads the attrition-model notebook they submitted; the model generates
a quiz grounded in that specific notebook (some questions are dynamic follow-ups
generated from the candidate's own live answer); the candidate answers one question
at a time while a stopwatch (not a countdown — there is no time limit) tracks
elapsed time; correct answers are never revealed until the final question breakdown.

One page of the multipage app — run the app via `streamlit run app.py`.
Auth: set OPENAI_API_KEY (e.g. in a local .env file).
"""

import json
import random
import time

import nbformat
import openai
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from db import get_model_followup, save_model_followup, save_quiz_result

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
Python package. Install via: pip install vivainsights. \
Note: this dataset is known to be incomplete. An attrition label \
column is missing and will need to be engineered or simulated — identifying and \
addressing this gap is part of the task.
The notebook should include: the recommended choice of attrition model algorithm \
and rationale; code demonstrating how the model would run end-to-end; example \
outputs (predictions, feature importance, evaluation metrics); and a brief \
discussion of production considerations."
"""

# The four models a candidate can be asked about. Fixed rather than LLM-invented so
# the model_followup question bank (below) has a small, known key space.
FIXED_MODEL_OPTIONS = ["Random Forest", "XGBoost", "Logistic Regression", "Support Vector Machine"]

# Fixed distractors for the "io" question -- the LLM only has to generate the one
# correct option, in the same "Input: ...; Output: ..." format.
FIXED_IO_DISTRACTORS = [
    "Input: employee workplace and activity features; Output: the feature importance values used to train the model",
    "Input: each employee's known future attrition status; Output: the workplace features most associated with leaving",
    "Input: employee identifiers and activity features; Output: the predicted date on which each employee will leave",
]

# The 4 questions generated together in one batch call (generate_quiz). label_exists
# is a separate, fixed (non-LLM) question inserted right after model_choice -- see
# label_exists_question(). model_followup and label_followup are dynamic follow-ups,
# materialized live during the quiz; see FOLLOWUP_AFTER / FOLLOWUP_SUBJECT_FOR_TRIGGER.
# This is the complete question set -- there are no other slots.
SLOT_ORDER = [
    "model_choice",
    "io",
    "metric_choice",
    "metric_interpretation",
]

# Where dynamic follow-ups are inserted relative to their trigger slot, and what
# "subject" the materialization logic should treat them as (see the quiz stage).
FOLLOWUP_AFTER = {
    "label_exists": "label_followup",
    "model_choice": "model_followup",
}
FOLLOWUP_SUBJECT_FOR_TRIGGER = {
    "label_exists": "label",
    "model_choice": "model",
}

SLOT_INSTRUCTIONS = {
    "model_choice": """\
1. model_choice — Determine which ONE of these four algorithms the notebook \
actually used/chose as its (primary or recommended) attrition model: "Random \
Forest", "XGBoost", "Logistic Regression", "Support Vector Machine". Set "options" \
to these four exact strings (verbatim, any order) and "correct_index" to whichever \
one the notebook actually used. If the notebook's actual model is not literally one \
of these four (e.g. Gradient Boosting, LightGBM, a neural network), pick whichever \
of the four is the closest conceptual match (other tree-boosting variants → \
XGBoost; other tree ensembles → Random Forest; other linear/regularized linear \
classifiers → Logistic Regression; other margin/kernel-based classifiers → Support \
Vector Machine) and name the notebook's actual model explicitly in the \
explanation.""",
    "io": """\
2. io — Determine the final model's actual inputs and outputs from the notebook, \
and phrase the correct option in EXACTLY this format: "Input: <...>; Output: \
<...>" (one short fragment per side). Set "options" to a list of exactly 4 \
strings: the 3 fixed distractors below, copied VERBATIM character-for-character, \
plus your one generated correct option — in any order — and set "correct_index" to \
point at your generated option.

Fixed distractors (copy verbatim, do not alter):
- "Input: employee workplace and activity features; Output: the feature importance values used to train the model"
- "Input: each employee's known future attrition status; Output: the workplace features most associated with leaving"
- "Input: employee identifiers and activity features; Output: the predicted date on which each employee will leave\"""",
    "metric_choice": """\
3. metric_choice — Ask: "How did the notebook evaluate the attrition models' \
performance?" All 4 options must use plausible, technical evaluation terminology \
(e.g., specific metric names, cross-validation, a held-out test set, a confusion \
matrix) and sound reasonable; exactly one must describe what the notebook actually \
did to evaluate performance, the other three must be plausible-but-wrong evaluation \
approaches it did NOT use.""",
    "metric_interpretation": """\
4. metric_interpretation — Identify the notebook's actual evaluation metric AND its \
actual reported value or range (e.g., "cross-validated AUC-ROC scores around \
0.53-0.58"). Quote that specific metric name and value/range in the question text, \
then ask what that result means. If the metric is AUC-ROC (or ROC-AUC): ground the \
correct answer in "only modestly better than random guessing (0.5), likely because \
the attrition labels were simulated with substantial noise rather than being real \
outcomes" — NOT a claim that the model or approach itself is broken. Style example \
(match this difficulty and structure, substituting the real metric/value from THIS \
notebook):

"The notebook reports that the models have cross-validated AUC-ROC scores around \
0.53-0.58. What is the best interpretation of this result?"
A. The models are performing perfectly and are ready for production deployment.
B. The models are only slightly better than random guessing, likely because the \
attrition labels were simulated with substantial noise.  [correct]
C. The models failed because AUC-ROC cannot be used for binary classification.
D. The models are inaccurate because the AUC-ROC score is low.

If the notebook used a DIFFERENT metric (not AUC-ROC), follow the same style: quote \
the real metric name and value/range from the notebook, with one correct \
technically-accurate interpretation and three plausible-but-wrong interpretations in \
the same spirit as the example (one "too good to be true" naive read, one factually \
wrong claim about what the metric measures or applies to, one oversimplified "the \
number is low so it's bad" claim that misses the actual nuance).""",
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

# Style exemplars for generate_model_followup: every option (correct AND
# distractors) shares similar structural/keyword framing, so the correct answer
# isn't identifiable just by noticing which option mentions the model's own
# characteristic keywords or is a different length/shape than the others.
MODEL_FOLLOWUP_FEWSHOT = """\
Style examples (the [correct] tag marks the right answer -- note every option, \
right or wrong, shares comparable structure and technical detail):

Logistic Regression:
A. It provides probabilities and interpretable coefficients, but nonlinear relationships may require additional feature engineering.  [correct]
B. It provides probabilities and interpretable coefficients, but only after combining predictions from many fitted trees.
C. It provides probabilities and nonlinear feature interactions automatically, but cannot show the direction of feature effects.
D. It provides probabilities and a flexible decision boundary, but usually requires extensive hyperparameter tuning.

XGBoost:
A. Its boosting procedure uses regularization and computational optimizations, but tuning and explaining the final model can require additional effort.  [correct]
B. Its boosting procedure trains all trees independently, but combining their predictions can require substantial memory.
C. Its boosting procedure captures nonlinear patterns, but requires every feature to be standardized and cannot handle missing values.
D. Its boosting procedure includes regularization, so overfitting and probability calibration do not need to be evaluated.
"""


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


def _reshuffle(q: QuizQuestion) -> None:
    """Shuffle a question's options in place, keeping correct_index pointing at the
    same option text. Used for slots with a fixed/partially-fixed option set, so the
    final order never depends on whatever order the LLM happened to emit."""
    correct_text = q.options[q.correct_index]
    order = list(range(len(q.options)))
    random.shuffle(order)
    q.options = [q.options[i] for i in order]
    q.correct_index = q.options.index(correct_text)


def label_exists_question() -> QuizQuestion:
    """A fixed, non-LLM question. The task's starter dataset (pq_data) is
    deliberately missing the attrition label for every candidate, so the correct
    answer never depends on the notebook's content -- no API call needed."""
    return QuizQuestion(
        slot="label_exists",
        topic="missing label",
        question="Is there an employee attrition label in the pq_data dataset used for this task?",
        options=["Yes", "No"],
        correct_index=1,
        explanation=(
            "pq_data is a known-incomplete starter dataset — it does not include a "
            "real attrition label. The notebook needed to engineer or simulate one "
            "before a model could be trained."
        ),
    )


BATCH_JSON_SCHEMA_NOTE = """\
Respond with ONLY a single JSON object (no markdown code fences, no commentary) \
matching this exact shape:
{"questions": [
  {"slot": "<one of: model_choice, io, metric_choice, metric_interpretation>",
   "topic": "<2-4 word tag>", "question": "<question text, may include \\n and \
fenced ```code``` blocks>",
   "options": ["<A>", "<B>", "<C>", "<D>"], "correct_index": <0-3 integer>,
   "explanation": "<1-3 sentence explanation>"}
]}
Return EXACTLY 4 questions, one per slot listed above, in that exact order \
(model_choice first, metric_interpretation last).
"""


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------

def generate_quiz(client: OpenAI, notebook_text: str) -> list[QuizQuestion]:
    """Generate the 4 static (non-follow-up) questions in one call."""
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
            if q.slot == "model_choice" and set(q.options) != set(FIXED_MODEL_OPTIONS):
                raise RuntimeError(
                    "The generated 'model_choice' options didn't match the required "
                    "fixed model list — try again."
                )
            if q.slot == "io":
                fixed_present = [o for o in q.options if o in FIXED_IO_DISTRACTORS]
                if len(fixed_present) != 3:
                    raise RuntimeError(
                        "The generated 'io' options didn't include the three "
                        "required fixed distractors — try again."
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

    # Python-side reshuffle guarantees true randomization for the two fixed-option
    # slots, regardless of what order the LLM happened to emit them in.
    for q in deduped:
        if q.slot in ("model_choice", "io"):
            _reshuffle(q)

    return deduped


def assemble_static_questions(client: OpenAI, notebook_text: str) -> list[QuizQuestion]:
    """Generate the 4 LLM-authored questions, then insert the fixed label_exists
    question right after model_choice."""
    generated = generate_quiz(client, notebook_text)
    return [generated[0], label_exists_question()] + generated[1:]


def _call_single_question(client: OpenAI, notebook_text: str, instructions: str) -> QuizQuestion:
    """Shared plumbing for the dynamic (single-question) follow-up generators."""
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


def generate_label_followup(client: OpenAI, notebook_text: str) -> QuizQuestion:
    """Follow-up asked only when the candidate correctly says the label is missing:
    how did THIS notebook actually address that gap."""
    instructions = f"""\
The candidate correctly identified that the dataset has no real attrition label. \
Write ONE follow-up multiple-choice question asking how THIS notebook actually \
addressed that gap (e.g., simulating a label from proxy behavioral signals, what \
signals it used, how it introduced randomness/noise, what threshold or heuristic it \
applied). Ground the correct option in exactly what the notebook did. The three \
distractors must use plausible, technical-sounding ML/data-prep terminology \
describing approaches the notebook did NOT take (e.g., a different simulation \
method, a wrong labeling source, an approach that wouldn't actually work) — not \
generic or obviously-wrong options.

{SHARED_REQUIREMENTS}
Respond with ONLY a single JSON object (no markdown code fences, no commentary) \
matching this exact shape:
{{"question": {{"slot": "label_followup", "topic": "<2-4 word tag>", \
"question": "<question text>", "options": ["<A>", "<B>", "<C>", "<D>"], \
"correct_index": <0-3 integer>, "explanation": "<1-3 sentence explanation>"}}}}
"""
    return _call_single_question(client, notebook_text, instructions)


def _model_followup_question_text(model_name: str, personalized: bool) -> str:
    if personalized:
        return f'What are the benefits and restrictions of using **{model_name}**, the model you selected?'
    return (
        f'What are the benefits and restrictions of using **{model_name}**, the '
        "model actually used in the notebook?"
    )


def generate_model_followup(
    client: OpenAI, notebook_text: str, model_name: str, personalized: bool
) -> QuizQuestion:
    """Benefits/restrictions question about `model_name`. Checks the shared
    Supabase-backed question bank first; only calls the LLM (and saves the result
    back to the bank) for a model not already stored there."""
    banked = get_model_followup(model_name)
    if banked is not None:
        return QuizQuestion(
            slot="model_followup",
            topic="model rationale",
            question=_model_followup_question_text(model_name, personalized),
            options=banked["options"],
            correct_index=banked["correct_index"],
            explanation=banked["explanation"],
        )

    instructions = f"""\
Write ONE multiple-choice question asking about the benefits AND restrictions of \
using "{model_name}" for this kind of attrition-prediction task. Exactly one option \
must be TRUE of "{model_name}"; the other three must be plausible-sounding but \
FALSE (true of a different algorithm, or subtly incorrect). Follow the style of the \
examples below closely: every option (correct and distractors alike) shares similar \
sentence structure and mentions a comparable structural/technical detail of its \
respective algorithm (trees, boosting, coefficients, kernels, etc.) paired with a \
"but ..." limitation — so the distractors can't be spotted just by noticing which \
option mentions "{model_name}"-specific keywords or is a different length.

{MODEL_FOLLOWUP_FEWSHOT}

{SHARED_REQUIREMENTS}
Respond with ONLY a single JSON object (no markdown code fences, no commentary) \
matching this exact shape:
{{"question": {{"slot": "model_followup", "topic": "<2-4 word tag>", \
"question": "<question text>", "options": ["<A>", "<B>", "<C>", "<D>"], \
"correct_index": <0-3 integer>, "explanation": "<1-3 sentence explanation>"}}}}
"""
    q = _call_single_question(client, notebook_text, instructions)
    save_model_followup(model_name, q.options, q.correct_index, q.explanation)
    q.question = _model_followup_question_text(model_name, personalized)
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
    """Interleave the static questions with follow-up placeholders, producing the
    full quiz sequence in template order."""
    items = []
    for q in static_questions:
        items.append(q)
        if q.slot in FOLLOWUP_AFTER:
            items.append({"placeholder": True, "subject": FOLLOWUP_SUBJECT_FOR_TRIGGER[q.slot]})
    return items


def _skipped_question(slot: str, reason: str) -> QuizQuestion:
    """A trivial placeholder for a follow-up that won't be generated (either the
    trigger's premise didn't hold, or generation failed and the candidate chose to
    skip). Excluded from scoring via the `topic == "skipped"` check in finish_quiz."""
    return QuizQuestion(
        slot=slot, topic="skipped",
        question=f"_({reason})_",
        options=["N/A", "N/A", "N/A", "N/A"], correct_index=0,
        explanation=reason,
    )


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
    placeholder, and any question explicitly marked as skipped)."""
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
        "model** task. A quiz (some questions generated live from your own answers) "
        "will check your understanding of your own submission. A stopwatch tracks "
        "how long you take, but there's no time limit."
    )

    candidate_name = st.text_input("Candidate name (optional, stored with your results)")
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
                assemble_static_questions, get_client(), notebook_text
            )
        if static_questions is not None:
            items = build_items(static_questions)
            st.session_state.quiz_items = items
            st.session_state.answers = [None] * len(items)
            st.session_state.time_spent = [None] * len(items)
            st.session_state.question_start_times = {}
            st.session_state.notebook_text = notebook_text
            st.session_state.candidate_name = candidate_name.strip() or None
            st.session_state.notebook_filename = uploaded.name
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
            # The follow-up's premise ("there is no label") only holds if the
            # candidate correctly answered "No" -- otherwise asking it doesn't make
            # sense, so skip it rather than generating a question on a false premise.
            st.session_state.quiz_items[idx] = _skipped_question(
                "label_followup",
                'Skipped — only asked when the previous question is answered "No".',
            )
            item = st.session_state.quiz_items[idx]
        else:
            with st.spinner("Generating a follow-up based on your answer…"):
                if subject == "label":
                    followup = call_with_errors_surfaced(
                        generate_label_followup, get_client(), st.session_state.notebook_text,
                    )
                else:  # "model"
                    model_name = chosen_label or trigger_item.options[trigger_item.correct_index]
                    followup = call_with_errors_surfaced(
                        generate_model_followup, get_client(), st.session_state.notebook_text,
                        model_name, chosen_label is not None,
                    )
            if followup is None:
                col_a, col_b = st.columns(2)
                if col_a.button("Retry"):
                    st.rerun()
                if col_b.button("Skip this question"):
                    st.session_state.quiz_items[idx] = _skipped_question(
                        f"{subject}_followup",
                        "This follow-up could not be generated and was skipped.",
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
        for q, a, t in zip(questions, answers, time_spent)
    ]
    results_payload = {
        "score": score,
        "total": total,
        "elapsed_seconds": round(st.session_state.elapsed, 1),
        "questions": question_records,
    }

    if "db_save_attempted" not in st.session_state:
        st.session_state.db_save_attempted = True
        db_payload = {
            "candidate_name": st.session_state.get("candidate_name"),
            "notebook_filename": st.session_state.get("notebook_filename"),
            "score": score,
            "total": total,
            "elapsed_seconds": round(st.session_state.elapsed, 1),
            "questions": question_records,
        }
        saved, db_error = save_quiz_result(db_payload)
        st.session_state.db_save_ok = saved
        st.session_state.db_save_error = db_error

    if st.session_state.get("db_save_ok"):
        st.caption("✅ Saved to database")
    else:
        st.caption(f"⚠️ Not saved to database: {st.session_state.get('db_save_error')}")

    st.download_button(
        "Download results (JSON)",
        data=json.dumps(results_payload, indent=2),
        file_name="quiz_results.json",
        mime="application/json",
    )
    st.button("Start over", on_click=reset)
