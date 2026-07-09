"""Notebook Quiz — a timed comprehension quiz generated from an uploaded Jupyter notebook.

A candidate uploads the attrition-model notebook they submitted; Claude generates
multiple-choice questions grounded in that specific notebook; the candidate answers
under a countdown; answers are scored against the generated key and Claude writes
an evaluator-facing report.

Run with:  streamlit run app.py
Auth:      set ANTHROPIC_API_KEY, or run `ant auth login` first.
"""

import json
import time

import anthropic
import nbformat
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

MODEL = "claude-opus-4-8"
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

QUESTION_TEMPLATE = """\
Use these template questions as calibration anchors. Every quiz you generate must \
match their difficulty, style, and topic blueprint — only the notebook-specific \
details (actual signals, models, metric values) change between notebooks.

Template question 1 (topic: missing attrition label):
"How does the notebook handle the missing attrition label?"
A. It removes all rows without attrition labels
B. It manually labels employees using random guessing only
C. It simulates attrition labels using behavioral signals plus noise  [correct]
D. It uses SHAP values as the attrition label
Explanation: Since the dataset did not contain a real attrition label, the notebook
creates a simulated label from proxy signals of employee risk (e.g., after-hours
work, manager interaction, collaboration patterns, meeting load). This matters
because the model only demonstrates what an attrition pipeline could look like —
production would need real HR attrition outcomes.

Template question 2 (topic: model comparison):
"Which models does the notebook compare?"
A. Linear regression, K-means, and PCA
B. Logistic Regression, Random Forest, and XGBoost  [correct]
C. Naive Bayes, SVM, and ARIMA
D. Decision Tree, LSTM, and DBSCAN
Explanation: Logistic Regression is a simple, interpretable linear model; Random
Forest is a tree ensemble capturing nonlinear patterns but less interpretable;
XGBoost builds trees sequentially to correct previous errors, often stronger on
tabular data but needing more tuning.

Template question 3 (topic: model outputs):
"What does the final model output include?"
— four options where the correct one names the notebook's actual output (e.g., a
per-employee attrition risk score), and distractors name outputs the notebook does
not produce.

Template question 4 (topic: metric interpretation):
"What does an AUC-ROC score around 0.53-0.58 suggest in this notebook?"
A. The model is nearly perfect
B. The model is worse than random guessing
C. The model has only a modest signal above random guessing  [correct]
D. The model proves the product is production-ready
Explanation: AUC-ROC measures how well the model ranks employees who attrite above
those who do not, across all thresholds; a value just above 0.5 means only modest
signal. Substitute the metric and value actually reported in the notebook.

Template question 5 (topic: explainability):
"What is the purpose of SHAP?"
A. It identifies which features contributed most to an employee's predicted attrition risk  [correct]
B. It selects the best classification threshold by maximizing recall
C. It converts weekly employee records into person-level features
D. It creates the simulated attrition label from behavioral signals
Explanation: SHAP attributes each prediction to feature contributions. Substitute
whatever explainability/feature-importance method the notebook actually uses.

Difficulty calibration: exactly one clearly correct option; distractors are
plausible because they mix up related concepts from the same notebook (e.g., using
SHAP as a label, or confusing preprocessing with explainability), but are clearly
wrong to someone who understands the work. Simple factual recall of the notebook's
own choices — not tricky edge cases, not obscure theory.
"""

SYSTEM_PROMPT = (
    "You are an expert technical interviewer for data science roles. You evaluate "
    "whether a candidate genuinely understands a take-home notebook they submitted "
    "— including work they may have produced with AI assistance. You ask precise, "
    "notebook-specific questions and give calibrated, evidence-based assessments.\n\n"
    + TASK_CONTEXT
)


class QuizQuestion(BaseModel):
    topic: str
    question: str
    options: list[str]
    correct_index: int
    explanation: str


class Quiz(BaseModel):
    questions: list[QuizQuestion]


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


def notebook_content_block(notebook_text: str) -> dict:
    """The notebook as a cacheable content block, shared by both API calls.

    Both calls use the same system prompt and lead with this exact block, so the
    grading call reads the prompt cache written by the generation call.
    """
    return {
        "type": "text",
        "text": f"<notebook>\n{notebook_text}\n</notebook>",
        "cache_control": {"type": "ephemeral"},
    }


# ---------------------------------------------------------------------------
# Claude calls
# ---------------------------------------------------------------------------

def generate_quiz(client: anthropic.Anthropic, notebook_text: str, n_questions: int) -> Quiz:
    instructions = f"""\
{QUESTION_TEMPLATE}

Generate exactly {n_questions} multiple-choice questions that test whether the \
person taking this quiz genuinely understands the notebook above (which they \
claim to have authored). Requirements:

- Follow the template blueprint in order: missing-label handling, model comparison \
/ algorithm choice, model outputs, metric interpretation, explainability. If \
{n_questions} exceeds the blueprint, add questions on production considerations \
and feature engineering at the same difficulty; if the notebook does not cover a \
blueprint topic, substitute another topic the notebook does cover.
- Ground every question in the specific content of THIS notebook: reference its \
actual modeling choices, engineered attrition label, feature names, metric values, \
and stated production considerations. Where the template's specifics differ from \
this notebook, the notebook wins.
- Exactly 4 options per question, exactly one correct (correct_index is 0-based), \
at the template's difficulty level.
- Each question must be answerable in about 45 seconds without re-running code.
- Do NOT ask trivia (import order, variable names, library versions).
- topic: a 2-4 word tag. explanation: 1-3 sentences on why the correct answer is \
right (in the style of the template explanations), written for the evaluator's \
report.
"""
    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        messages=[{
            "role": "user",
            "content": [
                notebook_content_block(notebook_text),
                {"type": "text", "text": instructions},
            ],
        }],
        output_format=Quiz,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to process this notebook (refusal stop reason).")
    quiz = response.parsed_output
    if quiz is None or not quiz.questions:
        raise RuntimeError("Quiz generation returned no parseable questions — try again.")
    # Drop malformed questions rather than rendering a broken quiz.
    quiz.questions = [
        q for q in quiz.questions
        if len(q.options) == 4 and 0 <= q.correct_index < 4
    ]
    if not quiz.questions:
        raise RuntimeError("All generated questions were malformed — try again.")
    return quiz


def generate_report(
    client: anthropic.Anthropic,
    notebook_text: str,
    quiz: Quiz,
    answers: list,
    elapsed_seconds: float,
) -> str:
    results = []
    for q, a in zip(quiz.questions, answers):
        results.append({
            "topic": q.topic,
            "question": q.question,
            "options": q.options,
            "correct_answer": q.options[q.correct_index],
            "candidate_answer": q.options[a] if a is not None else None,
            "answered_correctly": a == q.correct_index,
            "why_correct": q.explanation,
        })
    score = sum(r["answered_correctly"] for r in results)
    instructions = f"""\
The candidate just completed a timed comprehension quiz about the notebook above. \
Score: {score}/{len(results)}. Time used: {elapsed_seconds:.0f} seconds. \
Full results:

{json.dumps(results, indent=2)}

Write a concise evaluator-facing report in markdown (under 400 words) with:
1. **Overall assessment** — does the candidate appear to genuinely understand \
their submission? Unanswered questions usually mean they ran out of time.
2. **Understanding by topic** — where they showed command vs. gaps, referencing \
specific wrong answers and what the mistake suggests.
3. **Suggested follow-ups** — 2-3 targeted questions for a live conversation, \
aimed at the weakest areas.
Do not restate every question; synthesize.
"""
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        messages=[{
            "role": "user",
            "content": [
                notebook_content_block(notebook_text),
                {"type": "text", "text": instructions},
            ],
        }],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to grade this quiz (refusal stop reason).")
    return "".join(b.text for b in response.content if b.type == "text")


def call_with_errors_surfaced(fn, *args, **kwargs):
    """Run an API-calling function, converting SDK errors to readable messages."""
    try:
        return fn(*args, **kwargs)
    except anthropic.AuthenticationError:
        st.error(
            "Authentication failed. Set the ANTHROPIC_API_KEY environment variable "
            "(or run `ant auth login`) and restart the app."
        )
    except anthropic.RateLimitError:
        st.error("Rate limited by the API — wait a moment and try again.")
    except anthropic.APIStatusError as e:
        st.error(f"API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        st.error("Could not reach the Anthropic API — check your network connection.")
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
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def submit_quiz():
    quiz: Quiz = st.session_state.quiz
    answers = []
    for i, q in enumerate(quiz.questions):
        selected = st.session_state.get(f"q{i}")
        answers.append(q.options.index(selected) if selected in q.options else None)
    st.session_state.answers = answers
    st.session_state.elapsed = min(
        st.session_state.quiz_seconds, time.time() - st.session_state.started_at
    )
    st.session_state.stage = "results"


st.title("📝 Notebook Comprehension Quiz")

# ---- Stage: upload -------------------------------------------------------
if st.session_state.stage == "upload":
    st.markdown(
        "Upload the Jupyter notebook you submitted for the **employee attrition "
        "model** task. A short timed quiz will be generated from its contents to "
        "check your understanding of your own submission."
    )
    with st.sidebar:
        st.header("Settings")
        n_questions = st.slider("Number of questions", 3, 10, 6)
        minutes = st.slider("Time limit (minutes)", 1, 15, 5)

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
            quiz = call_with_errors_surfaced(
                generate_quiz, get_client(), notebook_text, n_questions
            )
        if quiz is not None:
            st.session_state.quiz = quiz
            st.session_state.notebook_text = notebook_text
            st.session_state.quiz_seconds = minutes * 60
            st.session_state.started_at = time.time()
            st.session_state.deadline = time.time() + minutes * 60
            st.session_state.stage = "quiz"
            st.rerun()

# ---- Stage: quiz ---------------------------------------------------------
elif st.session_state.stage == "quiz":
    quiz: Quiz = st.session_state.quiz

    if st.session_state.get("time_up"):
        submit_quiz()
        st.rerun()

    @st.fragment(run_every=1.0)
    def countdown():
        remaining = st.session_state.deadline - time.time()
        if remaining <= 0:
            st.session_state.time_up = True
            st.rerun(scope="app")
        mm, ss = divmod(int(max(remaining, 0)), 60)
        st.metric("⏱️ Time remaining", f"{mm}:{ss:02d}")
        if remaining <= 60:
            st.warning("Less than a minute left — unanswered questions score zero.")

    countdown()
    st.caption("The quiz auto-submits when the timer runs out.")
    st.divider()

    for i, q in enumerate(quiz.questions):
        st.subheader(f"Question {i + 1} of {len(quiz.questions)}")
        st.markdown(q.question)
        st.radio(
            "Select one:",
            q.options,
            index=None,
            key=f"q{i}",
            label_visibility="collapsed",
        )
        st.divider()

    if st.button("Submit answers", type="primary"):
        submit_quiz()
        st.rerun()

# ---- Stage: results ------------------------------------------------------
elif st.session_state.stage == "results":
    quiz: Quiz = st.session_state.quiz
    answers = st.session_state.answers
    score = sum(1 for q, a in zip(quiz.questions, answers) if a == q.correct_index)
    total = len(quiz.questions)

    col1, col2 = st.columns(2)
    col1.metric("Score", f"{score} / {total}")
    col2.metric("Time used", f"{st.session_state.elapsed:.0f}s")

    if "report" not in st.session_state:
        with st.spinner("Grading and writing the evaluator report…"):
            report = call_with_errors_surfaced(
                generate_report,
                get_client(),
                st.session_state.notebook_text,
                quiz,
                answers,
                st.session_state.elapsed,
            )
        if report is not None:
            st.session_state.report = report

    if "report" in st.session_state:
        st.markdown("## Evaluator report")
        st.markdown(st.session_state.report)

    st.markdown("## Question breakdown")
    for i, (q, a) in enumerate(zip(quiz.questions, answers)):
        correct = a == q.correct_index
        icon = "✅" if correct else ("⬜" if a is None else "❌")
        with st.expander(f"{icon} Q{i + 1} · {q.topic}"):
            st.markdown(q.question)
            st.markdown(f"- **Your answer:** {q.options[a] if a is not None else '_unanswered_'}")
            st.markdown(f"- **Correct answer:** {q.options[q.correct_index]}")
            st.markdown(f"- **Why:** {q.explanation}")

    results_payload = {
        "score": score,
        "total": total,
        "elapsed_seconds": round(st.session_state.elapsed, 1),
        "report": st.session_state.get("report"),
        "questions": [
            {
                "topic": q.topic,
                "question": q.question,
                "options": q.options,
                "correct_index": q.correct_index,
                "candidate_answer_index": a,
                "answered_correctly": a == q.correct_index,
                "explanation": q.explanation,
            }
            for q, a in zip(quiz.questions, answers)
        ],
    }
    st.download_button(
        "Download results (JSON)",
        data=json.dumps(results_payload, indent=2),
        file_name="quiz_results.json",
        mime="application/json",
    )
    st.button("Start over", on_click=reset)
