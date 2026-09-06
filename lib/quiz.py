"""Notebook comprehension quiz — generation, blind self-check, and dynamic follow-ups.

A quiz is generated from a specific uploaded notebook (some questions are dynamic
follow-ups generated from the participant's own live answer). Quiz generation runs on
Claude Opus 5; every LLM-authored question is then *self-checked* — the model takes the
freshly written quiz blind (options only, no answer key) and reports which options are
defensibly correct. Any question that doesn't have exactly one defensible answer — its
own designated one — is sent back for a minimal repair, then re-checked. See
verify_quiz().

This module holds only the generation/verification logic (no Streamlit UI). The
in-app quiz-taking flow lives in lib/quiz_ui.py.
Auth: set ANTHROPIC_API_KEY (e.g. in a local .env file).
"""

import json
import random
import re

import anthropic
import streamlit as st
from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

from db import get_model_followup, save_model_followup

MODEL = "claude-opus-5"  # swap to "claude-sonnet-5" for cheaper/faster test iteration

# How many repair→re-check rounds a flagged question gets before we give up and
# just record a warning.
VERIFY_ROUNDS = 2

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
# the model_followup question bank has a small, known key space.
FIXED_MODEL_OPTIONS = ["Random Forest", "XGBoost", "Logistic Regression", "Support Vector Machine"]

# Slots whose option SET is fixed (only correct_index may move during a repair).
FIXED_OPTION_SLOTS = {"model_choice", "io_overlap"}

# Hardcoded distractor pools. For these slots the distractors are taken VERBATIM
# from the pool (three that the notebook did NOT do); the correct option is a pool
# entry when one fits and otherwise a model-written sentence held to the pool's
# length and specificity. This keeps distractor quality and detail level consistent
# across notebooks and stops the correct answer from standing out by being the most
# specific option. generate_quiz snaps each banked question's options back onto the
# pool (see _snap_banked_options).
DISTRACTOR_BANK: dict[str, list[str]] = {
    "aggregation": [
        "Grouped rows by employee and averaged each metric across all 35 weekly rows, producing one row per employee.",
        "Grouped rows by employee and summed each metric across all 35 weekly rows, producing one row per employee.",
        "Grouped rows by employee and kept only the most recent week's row, discarding the earlier weeks.",
        "Grouped rows by employee and used a trailing multi-week average ending at the most recent week.",
        "Grouped rows by employee and took the median of each metric across all 35 weekly rows.",
        "Grouped rows by employee and used the slope of a linear trend fitted to each metric over the 35 weeks.",
        "Did not aggregate: kept all 35 weekly rows per employee as separate training examples.",
    ],
    "train_test_split": [
        "A random 80/20 train/test split using a fixed random seed.",
        "A random 70/30 train/test split, stratified on the attrition label.",
        "A chronological split: earlier weeks for training, the latest weeks held out for testing.",
        "5-fold cross-validation with no separate held-out test set.",
        "A grouped split by employee, so no employee appears in both training and test.",
        "A random 90/10 train/test split with no fixed seed.",
    ],
    "inputs": [
        "The per-employee aggregated behavioural metrics from Viva Insights.",
        "The behavioural metrics together with the attrition label or the score used to build it.",
        "Employee and organisational identifiers rather than behavioural metrics.",
        "The un-aggregated weekly rows fed to the model as-is.",
        "A small subset of metrics hand-picked by their correlation with the attrition label.",
    ],
    "outputs": [
        "A predicted probability of attrition per employee (with an implied yes/no class).",
        "A ranked list of feature importances showing which behaviours drive attrition.",
        "A predicted number of weeks until each employee is expected to leave.",
        "A predicted value of a workplace metric (such as after-hours hours) used as a burnout proxy.",
        "A low / medium / high risk-tier label rather than a probability or class.",
    ],
    "metric_choice": [
        "Cross-validated ROC-AUC on the training set, reported as a mean across folds.",
        "Accuracy and a confusion matrix on a held-out test set.",
        "Precision, recall and F1 at a 0.5 threshold on a held-out test set.",
        "Training-vs-test log-loss compared to check for overfitting.",
        "A single train/test ROC-AUC with no cross-validation.",
        "A calibration curve comparing predicted probabilities to observed attrition rates.",
    ],
}
BANKED_SLOTS = set(DISTRACTOR_BANK)

SLOT_ORDER = [
    "aggregation",
    "train_test_split",
    "model_choice",
    "io_overlap",
    "inputs",
    "outputs",
    "metric_choice",
    "metric_interpretation",
]

# Where dynamic follow-ups are inserted relative to their trigger slot, and what
# "subject" the materialization logic should treat them as.
FOLLOWUP_AFTER = {
    "label_exists": "label_followup",
    "model_choice": "model_followup",
}
FOLLOWUP_SUBJECT_FOR_TRIGGER = {
    "label_exists": "label",
    "model_choice": "model",
}

SLOT_INSTRUCTIONS = {
    "aggregation": """\
1. aggregation — Ask: "How did the notebook aggregate the dataset's rows (which \
start as one row per employee per week) before using them to train the model?" \
This slot has a POOL (below). Use exactly 3 POOL entries VERBATIM as the distractors \
(pick ones the notebook did NOT do), and for the correct option use the POOL entry \
that matches the notebook verbatim -- or, only if none fits, one sentence of your \
own at the same length and detail as the POOL entries.""",
    "train_test_split": """\
2. train_test_split — Ask: "How did the notebook split the data into training and \
test sets?" This slot has a POOL (below). Use exactly 3 POOL entries VERBATIM as the \
distractors (ones the notebook did NOT do), and for the correct option use the POOL \
entry that matches the notebook verbatim -- or, only if none fits, one sentence of \
your own at the same length and detail as the POOL entries.""",
    "model_choice": """\
3. model_choice — Determine which ONE of these four algorithms the notebook \
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
    "io_overlap": """\
4. io_overlap — Ask exactly: "Are there any variables that were in both the input \
and output of your model?" Determine, from the notebook, whether ANY \
variable/feature used as a model INPUT also appears as (or leaks into) the \
model's OUTPUT/target. Set "options" to exactly ["Yes", "No"] (this slot uses only \
2 options, not 4) and "correct_index" to whichever is actually true of the \
notebook's final model — the option "Yes" if there is overlap/leakage, "No" if the \
inputs and output are cleanly separated. Answer based on what the notebook's final \
model actually does, not a hypothetical.""",
    "inputs": """\
5. inputs — Ask: "What are the inputs (features) of the final model?" This slot has \
a POOL (below). Use exactly 3 POOL entries VERBATIM as the distractors (ones that \
are false for this notebook), and for the correct option use the POOL entry that \
matches the notebook -- or, only if none fits, one sentence of your own at the same \
length and detail as the POOL entries. Do NOT list the notebook's exact column \
names; describe the feature set at the POOL's level of generality.""",
    "outputs": """\
6. outputs — Ask: "What is the output (target) of the final model?" This slot has a \
POOL (below). Use exactly 3 POOL entries VERBATIM as the distractors (ones that are \
false for this notebook), and for the correct option use the POOL entry that matches \
the notebook -- or, only if none fits, one sentence of your own at the same length \
and detail as the POOL entries.""",
    "metric_choice": """\
7. metric_choice — Ask: "How did the notebook evaluate the attrition model's \
performance?" This slot has a POOL (below). Use exactly 3 POOL entries VERBATIM as \
the distractors (evaluation approaches the notebook did NOT use), and for the \
correct option use the POOL entry that matches the notebook -- or, only if none \
fits, one sentence of your own at the same length and detail as the POOL entries.""",
    "metric_interpretation": """\
8. metric_interpretation — Identify the notebook's actual evaluation metric AND its \
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
- Exactly 4 options per question (except "io_overlap", which uses exactly 2: \
"Yes"/"No"), exactly one correct (correct_index is 0-based).
- Ground every question in the specific content of THIS notebook: reference its \
actual modeling choices, engineered attrition label, feature names, metric values, \
and stated production considerations.
- Each question must be answerable in about 45-60 seconds without re-running code.
- Do NOT ask trivia (import order, variable names, library versions).
- topic: a 2-4 word tag. explanation: 1-3 sentences on why the correct answer is \
right, shown to the candidate in the question breakdown after they finish.
- Distractors must be plausible — they mix up related concepts from the same \
notebook — but clearly wrong to someone who understands the work. Critically, \
exactly ONE option may be defensibly correct; a distractor that is also arguably \
true makes the question unusable.
- The correct option must NOT stand out. Match the distractors' length, sentence \
structure, and level of technical detail exactly. Never put a notebook-specific \
identifier (an exact column name, random seed, row/employee count, or metric value) \
in the correct option unless the distractors carry the same kind of detail. A \
correct answer a reader could pick just because it is the most specific or detailed \
option is a failed question — the correct answer is often the one that ends up too \
specific, so keep it general.
- Each distractor must be a genuinely different approach from the correct answer, \
not a near-paraphrase of it.
- For a slot that provides a POOL, the three distractors are copied VERBATIM from \
that POOL (never invented), and the correct option is a POOL entry whenever one \
fits the notebook.
"""

SYSTEM_PROMPT = (
    "You are an expert technical interviewer for data science roles. You evaluate "
    "whether a candidate genuinely understands a take-home notebook they submitted "
    "— including work they may have produced with AI assistance. You ask precise, "
    "notebook-specific questions and give calibrated, evidence-based assessments.\n\n"
    + TASK_CONTEXT
)

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


class QuestionVerdict(BaseModel):
    """One entry from a blind "take the quiz" pass."""

    slot: str
    defensible_option_indices: list[int]
    well_formed: bool
    issue: str


class QuizVerification(BaseModel):
    verdicts: list[QuestionVerdict]


# ---------------------------------------------------------------------------
# Notebook wrapping for the prompt
# ---------------------------------------------------------------------------

def notebook_prompt_prefix(notebook_text: str) -> str:
    """The notebook wrapped for the prompt. Every model call leads with this exact
    block and it carries a `cache_control` breakpoint (see _parse_call), so
    Anthropic prompt caching applies across the batch, verify, repair, and
    follow-up calls for one notebook."""
    return f"<notebook>\n{notebook_text}\n</notebook>"


def _reshuffle(q: QuizQuestion) -> None:
    """Shuffle a question's options in place, keeping correct_index pointing at the
    same option text."""
    correct_text = q.options[q.correct_index]
    order = list(range(len(q.options)))
    random.shuffle(order)
    q.options = [q.options[i] for i in order]
    q.correct_index = q.options.index(correct_text)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")


def _slot_instruction(slot: str) -> str:
    """The per-slot instruction, with its distractor POOL appended for banked slots."""
    text = SLOT_INSTRUCTIONS[slot]
    pool = DISTRACTOR_BANK.get(slot)
    if pool:
        text += "\n\nPOOL:\n" + "\n".join(f"- {p}" for p in pool)
    return text


def _snap_banked_options(q: QuizQuestion) -> None:
    """Force a banked slot's options onto its pool: keep the model's correct option,
    fill the other three slots with pool entries the model already picked (snapped to
    their canonical text) plus, if short, unused pool entries. Mutates in place."""
    pool = DISTRACTOR_BANK.get(q.slot)
    if not pool:
        return
    canon = {_norm(p): p for p in pool}
    correct = q.options[q.correct_index] if 0 <= q.correct_index < len(q.options) else pool[0]
    correct = canon.get(_norm(correct), correct)  # snap the correct option too if it's a pool entry

    distractors: list[str] = []
    for opt in q.options:
        c = canon.get(_norm(opt))
        if c and c != correct and c not in distractors:
            distractors.append(c)
    for p in pool:
        if len(distractors) >= 3:
            break
        if _norm(p) != _norm(correct) and p not in distractors:
            distractors.append(p)

    q.options = [correct, *distractors[:3]]
    q.correct_index = 0


def label_exists_question() -> QuizQuestion:
    """A fixed, non-LLM question. pq_data is deliberately missing the attrition label
    for every candidate, so the correct answer never depends on the notebook."""
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


def num_employees_question() -> QuizQuestion:
    """A fixed, non-LLM question. pq_data always has 300 unique PersonId values."""
    q = QuizQuestion(
        slot="num_employees",
        topic="dataset size",
        question="How many employees are in the pq_data dataset used for this task?",
        options=["100", "300", "1000", "3000"],
        correct_index=1,
        explanation=(
            "pq_data contains 300 unique employees (distinct PersonId values), each "
            "with one row per week over a 35-week period."
        ),
    )
    _reshuffle(q)
    return q


def row_granularity_question() -> QuizQuestion:
    """A fixed, non-LLM question about pq_data's row grain."""
    q = QuizQuestion(
        slot="row_granularity",
        topic="row granularity",
        question=(
            "What does a single row represent in the original pq_data dataset "
            "(before any aggregation)?"
        ),
        options=[
            "One employee's metrics",
            "One employee's metrics per week",
            "One employee's metrics per month",
            "One metric per employee",
        ],
        correct_index=1,
        explanation=(
            "pq_data is a weekly Person Query: each row is one employee's "
            "behavioral metrics for a single week (the MetricDate column), with 35 "
            "weekly rows per employee."
        ),
    )
    _reshuffle(q)
    return q


def io_overlap_ok_question() -> QuizQuestion:
    """A fixed, non-LLM question testing general ML knowledge (data leakage)."""
    return QuizQuestion(
        slot="io_overlap_ok",
        topic="data leakage",
        question=(
            "Is it okay for a variable to be used as both an input (feature) and "
            "the output (target/label) of a predictive model?"
        ),
        options=["Yes", "No"],
        correct_index=1,
        explanation=(
            "No — if a variable used to construct the label also appears among the "
            "input features, the model can 'leak' information about the label "
            "through that feature, producing misleadingly good performance that "
            "won't hold up on genuinely unseen data."
        ),
    )


BATCH_JSON_SCHEMA_NOTE = """\
Return EXACTLY 8 questions, one per slot listed above, in that exact order \
(aggregation first, metric_interpretation last). The "slot" of each question must \
be one of: aggregation, train_test_split, model_choice, io_overlap, inputs, \
outputs, metric_choice, metric_interpretation. Every slot uses exactly 4 options \
EXCEPT "io_overlap", which uses exactly 2 options: ["Yes", "No"] (either order). \
The "question" text may include newlines and fenced ```code``` blocks.
"""


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------

def _parse_call(
    client: Anthropic,
    instructions: str,
    notebook_text: str,
    schema: type[BaseModel],
    max_tokens: int = 32000,
):
    """One structured Claude call: system prompt + cached notebook + instructions,
    parsed into `schema`. Returns a validated `schema` instance or raises
    RuntimeError with a user-facing message."""
    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=max_tokens,
            # Explicit timeout: without it the SDK refuses a non-streaming call whose
            # max_tokens *could* run >10 min. These calls take well under that; 10 min
            # is a safe ceiling.
            timeout=600.0,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": notebook_prompt_prefix(notebook_text),
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": instructions},
                ],
            }],
            output_format=schema,
        )
    except ValidationError as e:
        raise RuntimeError(f"The model returned a malformed response: {e} — try again.") from e

    if response.stop_reason == "refusal":
        detail = getattr(response.stop_details, "explanation", None) or "safety refusal"
        raise RuntimeError(f"The model declined to process this notebook ({detail}).")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("The model ran out of room before finishing — try again.")

    parsed = response.parsed_output
    if parsed is None:
        raise RuntimeError("The model returned an unparseable response — try again.")
    return parsed


def generate_quiz(client: Anthropic, notebook_text: str) -> list[QuizQuestion]:
    """Generate the 8 LLM-authored, static (non-follow-up) questions in one call."""
    instructions = "\n\n".join(_slot_instruction(s) for s in SLOT_ORDER)
    instructions = f"{instructions}\n\n{SHARED_REQUIREMENTS}\n{BATCH_JSON_SCHEMA_NOTE}"

    quiz: Quiz = _parse_call(client, instructions, notebook_text, Quiz)

    order_index = {slot: i for i, slot in enumerate(SLOT_ORDER)}
    seen = set()
    deduped = []
    for q in quiz.questions:
        if q.slot in order_index and q.slot not in seen:
            expected_len = 2 if q.slot == "io_overlap" else 4
            if len(q.options) != expected_len or not (0 <= q.correct_index < expected_len):
                raise RuntimeError(
                    f"The generated '{q.slot}' question was malformed — try again."
                )
            if q.slot == "model_choice" and set(q.options) != set(FIXED_MODEL_OPTIONS):
                raise RuntimeError(
                    "The generated 'model_choice' options didn't match the required "
                    "fixed model list — try again."
                )
            if q.slot == "io_overlap" and set(q.options) != {"Yes", "No"}:
                raise RuntimeError(
                    "The generated 'io_overlap' options must be exactly Yes/No — "
                    "try again."
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

    for q in deduped:
        if q.slot in BANKED_SLOTS:
            _snap_banked_options(q)
        _reshuffle(q)  # every slot -- option position must never be a tell

    return deduped


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

TAKE_QUIZ_INSTRUCTIONS = """\
Below is a set of multiple-choice questions that were just written about THIS \
notebook, for a comprehension quiz its author will take. Take the quiz yourself, \
using ONLY the notebook as the source of truth — you are checking the questions, \
not the candidate.

For each question, fill in `defensible_option_indices`: every 0-based option index \
that a knowledgeable reader could correctly defend as a right answer, given what \
the notebook actually does. A usable question has EXACTLY ONE defensible answer:
  - If two or more options are each independently correct, list ALL of them.
  - If NO option is correct — e.g. the question asks about something the notebook \
never does — return an empty list.
Set `well_formed` to false, with a one-sentence `issue`, for any question that has \
zero or multiple defensible answers, is ambiguous, relies on information not in the \
notebook, or is otherwise poorly written. ALSO set `well_formed` to false if the \
correct option is noticeably longer, more specific, or more technically detailed \
than the distractors (a giveaway), or if any distractor is a near-paraphrase of the \
correct option rather than a genuinely different approach. Otherwise set \
`well_formed` to true and leave `issue` as "".

Return exactly one verdict per question, using the same `slot` values.

QUESTIONS:
%s
"""

REPAIR_INSTRUCTIONS = """\
Each question below was written about THIS notebook for the author's comprehension \
quiz, and a review found a problem: it does not have exactly one defensible correct \
answer, or it is not well formed. Rewrite each one so that EXACTLY ONE option is \
correct given the notebook.

Rules:
- Keep each question's `slot` and `topic`. Keep the question stem and the options \
as close to the originals as possible — make the SMALLEST change that fixes the \
problem.
- `intended_correct_index` is the answer the quiz is meant to test. Keep that \
option as the single correct answer whenever it is actually correct per the \
notebook. When the review shows other options are ALSO correct, minimally reword \
just those distractors so they become clearly incorrect while staying plausible \
and comparable to the correct option in length, structure, and specificity.
- Only if `intended_correct_index` is itself wrong per the notebook: fix that \
option's text, or (if another listed option is the true answer) set `correct_index` \
to it.
- Keep the same NUMBER of options. Do NOT change the option set for a Yes/No \
question, or for the model-choice question whose four options are fixed \
("Random Forest", "XGBoost", "Logistic Regression", "Support Vector Machine") — \
for those, only `correct_index` may change.
- For a slot with a `pool` field: fix an also-correct or paraphrased distractor by \
REPLACING it with a different `pool` entry the notebook did NOT do (verbatim) — do \
not reword it. Keep the correct option at the pool's length and level of detail.
- If the review flagged the correct option as too specific / a giveaway, make it \
LESS specific to match the distractors — do not make the distractors more specific.
- Keep `explanation` accurate for the final correct answer (1-3 sentences).

FLAGGED QUESTIONS:
%s

Return one corrected question per entry, keyed by the same `slot`, each in the full \
question shape (slot, topic, question, options, correct_index, explanation).
"""


def _take_quiz_blind(
    client: Anthropic, notebook_text: str, questions: list[QuizQuestion]
) -> dict[str, QuestionVerdict]:
    """Ask the model to answer `questions` from the notebook alone (no answer key)
    and report which options are defensible. Keyed by slot."""
    payload = json.dumps(
        [
            {"slot": q.slot, "question": q.question, "options": list(q.options)}
            for q in questions
        ],
        indent=2,
    )
    result: QuizVerification = _parse_call(
        client, TAKE_QUIZ_INSTRUCTIONS % payload, notebook_text, QuizVerification
    )
    return {v.slot: v for v in result.verdicts}


def _repair_questions(
    client: Anthropic,
    notebook_text: str,
    flagged: list[tuple[QuizQuestion, QuestionVerdict]],
) -> dict[str, QuizQuestion]:
    """Regenerate the flagged questions so each has exactly one correct option."""
    payload = json.dumps(
        [
            {
                "slot": q.slot,
                "topic": q.topic,
                "question": q.question,
                "options": list(q.options),
                "intended_correct_index": q.correct_index,
                "explanation": q.explanation,
                "review_defensible_indices": v.defensible_option_indices,
                "review_issue": v.issue,
                **({"pool": DISTRACTOR_BANK[q.slot]} if q.slot in BANKED_SLOTS else {}),
            }
            for q, v in flagged
        ],
        indent=2,
    )
    result: Quiz = _parse_call(
        client, REPAIR_INSTRUCTIONS % payload, notebook_text, Quiz
    )
    return {q.slot: q for q in result.questions}


def _accept_repair(original: QuizQuestion, repaired: QuizQuestion) -> QuizQuestion | None:
    """Return the repaired question only if it still satisfies this slot's
    structural constraints; otherwise None."""
    n = len(original.options)
    if len(repaired.options) != n or not (0 <= repaired.correct_index < n):
        return None
    if original.slot == "model_choice" and set(repaired.options) != set(FIXED_MODEL_OPTIONS):
        return None
    if original.slot == "io_overlap" and set(repaired.options) != {"Yes", "No"}:
        return None
    repaired.slot = original.slot
    repaired.topic = repaired.topic or original.topic
    return repaired


def verify_quiz(
    client: Anthropic,
    notebook_text: str,
    questions: list[QuizQuestion],
    rounds: int = VERIFY_ROUNDS,
) -> tuple[list[QuizQuestion], list[str]]:
    """Take the quiz blind and repair any question that doesn't have exactly one
    defensible answer. Returns the (possibly repaired) questions and a list of
    human-readable warnings for questions that couldn't be made single-answer."""
    questions = list(questions)
    idx_by_slot = {q.slot: i for i, q in enumerate(questions)}
    pending = list(range(len(questions)))
    warnings: list[str] = []

    for attempt in range(rounds + 1):
        if not pending:
            break
        verdicts = _take_quiz_blind(client, notebook_text, [questions[i] for i in pending])

        flagged: list[tuple[QuizQuestion, QuestionVerdict]] = []
        for i in pending:
            q = questions[i]
            v = verdicts.get(q.slot)
            if v is None:
                continue
            defensible = {d for d in v.defensible_option_indices if 0 <= d < len(q.options)}
            if v.well_formed and defensible == {q.correct_index}:
                continue
            if attempt == rounds:
                warnings.append(
                    f"{q.slot}: {v.issue or 'the self-check did not converge on a single correct answer'}"
                )
            else:
                flagged.append((q, v))

        if not flagged:
            break

        repaired = _repair_questions(client, notebook_text, flagged)
        next_pending: list[int] = []
        for q, _ in flagged:
            candidate = repaired.get(q.slot)
            candidate = _accept_repair(q, candidate) if candidate is not None else None
            if candidate is None:
                warnings.append(
                    f"{q.slot}: automated repair did not produce a valid replacement"
                )
                continue
            if candidate.slot in BANKED_SLOTS:
                _snap_banked_options(candidate)
            _reshuffle(candidate)
            questions[idx_by_slot[q.slot]] = candidate
            next_pending.append(idx_by_slot[q.slot])
        pending = next_pending

    return questions, warnings


def _note_warnings(warns: list[str]) -> None:
    """Accumulate self-check warnings on the session so the results page and the
    downloaded JSON can report them."""
    if warns:
        st.session_state.generation_warnings = (
            (st.session_state.get("generation_warnings") or []) + warns
        )


def _verified_single(
    client: Anthropic, notebook_text: str, q: QuizQuestion, rounds: int = 1
) -> QuizQuestion:
    """verify_quiz for one dynamic follow-up (fewer rounds — the candidate is
    waiting). Records any warnings on the session."""
    fixed, warns = verify_quiz(client, notebook_text, [q], rounds=rounds)
    _note_warnings(warns)
    return fixed[0]


def assemble_static_questions(client: Anthropic, notebook_text: str) -> list[QuizQuestion]:
    """Generate the 8 LLM-authored questions, self-check/repair them, then
    interleave the 4 fixed (non-LLM) questions at their fixed positions."""
    generated = generate_quiz(client, notebook_text)
    generated, warnings = verify_quiz(client, notebook_text, generated)
    st.session_state.generation_warnings = warnings

    by_slot = {q.slot: q for q in generated}
    return [
        num_employees_question(),
        row_granularity_question(),
        by_slot["aggregation"],
        label_exists_question(),
        by_slot["train_test_split"],
        by_slot["model_choice"],
        by_slot["io_overlap"],
        io_overlap_ok_question(),
        by_slot["inputs"],
        by_slot["outputs"],
        by_slot["metric_choice"],
        by_slot["metric_interpretation"],
    ]


def _call_single_question(client: Anthropic, notebook_text: str, instructions: str) -> QuizQuestion:
    """Shared plumbing for the dynamic (single-question) follow-up generators."""
    wrapped: SingleQuestion = _parse_call(
        client, instructions, notebook_text, SingleQuestion, max_tokens=8000
    )
    q = wrapped.question
    if len(q.options) != 4 or not (0 <= q.correct_index < 4):
        raise RuntimeError("The generated follow-up question was malformed — try again.")
    return q


def generate_label_followup(client: Anthropic, notebook_text: str) -> QuizQuestion:
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
generic or obviously-wrong options, and none of them defensibly correct.

{SHARED_REQUIREMENTS}
Produce one question with slot "label_followup", a 2-4 word topic, the question \
text, exactly 4 options, a 0-based correct_index, and a 1-3 sentence explanation.
"""
    q = _call_single_question(client, notebook_text, instructions)
    q = _verified_single(client, notebook_text, q)
    return q


def _model_followup_question_text(model_name: str, personalized: bool) -> str:
    if personalized:
        return f'What are the benefits and restrictions of using **{model_name}**, the model you selected?'
    return (
        f'What are the benefits and restrictions of using **{model_name}**, the '
        "model actually used in the notebook?"
    )


def generate_model_followup(
    client: Anthropic, notebook_text: str, model_name: str, personalized: bool
) -> QuizQuestion:
    """Benefits/restrictions question about `model_name`. Checks the shared
    Supabase-backed question bank first; only calls the LLM (self-checks it, and
    saves the checked result back to the bank) for a model not already stored."""
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
FALSE (true of a different algorithm, or subtly incorrect) — none of them \
defensibly correct. Follow the style of the examples below closely: every option \
(correct and distractors alike) shares similar sentence structure and mentions a \
comparable structural/technical detail of its respective algorithm (trees, \
boosting, coefficients, kernels, etc.) paired with a "but ..." limitation — so the \
distractors can't be spotted just by noticing which option mentions \
"{model_name}"-specific keywords or is a different length.

{MODEL_FOLLOWUP_FEWSHOT}

{SHARED_REQUIREMENTS}
Produce one question with slot "model_followup", a 2-4 word topic, the question \
text, exactly 4 options, a 0-based correct_index, and a 1-3 sentence explanation.
"""
    q = _call_single_question(client, notebook_text, instructions)
    q = _verified_single(client, notebook_text, q)
    save_model_followup(model_name, q.options, q.correct_index, q.explanation)
    q.question = _model_followup_question_text(model_name, personalized)
    return q


def call_with_errors_surfaced(fn, *args, **kwargs):
    """Run an API-calling function, converting SDK errors to readable messages."""
    try:
        return fn(*args, **kwargs)
    except anthropic.AuthenticationError:
        st.error(
            "Authentication failed. Set the ANTHROPIC_API_KEY environment variable "
            "(e.g. in a local .env file) and restart the app."
        )
    except anthropic.RateLimitError:
        st.error("Rate limited (or out of quota) — check your Anthropic plan/billing and try again.")
    except anthropic.APIStatusError as e:
        st.error(f"API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        st.error("Could not reach the Anthropic API — check your network connection.")
    except RuntimeError as e:
        st.error(str(e))
    return None
