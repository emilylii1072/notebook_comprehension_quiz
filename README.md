# Notebook Comprehension Quiz

A Streamlit webapp for dynamic evaluation of the **employee attrition model** take-home task.
The quiz-taker uploads the Jupyter notebook they submitted; an OpenAI model generates a fixed,
10-question quiz grounded in that specific notebook; answers are scored against the generated
key and the model writes an evaluator-facing report.

## How it works

1. **Upload** — the candidate uploads their `.ipynb`. The app flattens markdown, code,
   and (truncated) cell outputs into a transcript.
2. **Generate** — the model generates 8 questions in one batch call, following a fixed
   template (see below) so difficulty and coverage are consistent across notebooks and
   runs. JSON-mode output plus Pydantic validation guarantee well-formed questions.
3. **Quiz** — questions are presented one at a time under a countdown timer (default 5
   minutes); it auto-submits when time expires. Two questions (the model/metric
   follow-ups) are generated live, right after the candidate answers the model-choice
   and metric-choice questions, so they test *why* the candidate's own pick was
   appropriate — not a generic recall question. Unanswered questions score zero.
4. **Report** — the score is computed against the answer key, and the model writes a
   short evaluator report (overall assessment, understanding by topic, suggested live
   follow-up questions). Results — including per-question time spent — are downloadable
   as JSON.

## Question template (fixed, 10 slots in order)

1. **EDA code** — identify which of 4 candidate code lines (one real, verbatim from the
   notebook; three plausible fakes) they actually used, and why.
2. **Missing attrition label** — how the notebook addressed the missing label.
3. **Model choice** — which algorithm they used (one real option among plausible unused
   ones).
4. **Model follow-up** *(dynamically generated)* — why the algorithm they just picked
   would be appropriate for this task.
5. **Model training code** — a real snippet from the notebook's training section; pick
   the correct description of what it does.
6. **Evaluation metric** — which metric they used (one real option among plausible
   unused ones).
7. **Metric follow-up** *(dynamically generated)* — why the metric they just picked was
   an appropriate choice.
8. **Notebook outline** — identify the real section-by-section structure of the
   notebook among plausible-but-wrong orderings.
9. **Inputs/outputs** — what the final model actually takes in and produces.
10. **Production considerations** — what the notebook actually said about productionizing.

## Setup

```powershell
pip install -r requirements.txt
# create a .env file with: OPENAI_API_KEY = "sk-..."
python -m streamlit run app.py
```

The app opens at http://localhost:8501.

## Configuration

- **Time limit** (1–15 min, default 5) is in the sidebar on the upload screen. Question
  count is fixed at 10 (8 generated up front + 2 live follow-ups) to match the template.
- The per-slot instructions live in `SLOT_INSTRUCTIONS` in `app.py`; the task description
  shown to the model is in `TASK_CONTEXT`.
- `MODEL` in `app.py` defaults to `gpt-4o`; swap to `gpt-4o-mini` for cheaper/faster
  test iteration.

## Notes

- All API calls share an identical system prompt and lead with the same notebook text,
  so OpenAI's automatic prompt-prefix caching can apply across the batch, follow-up, and
  grading calls.
- The two follow-up questions (slots 4 and 7) are generated mid-quiz, based on the
  candidate's own answer to the preceding question — if they left that question
  unanswered, the follow-up falls back to asking about the notebook's actual choice
  instead of a personalized one.
- Every question records how long the candidate spent on it (from the moment it was
  shown); this is included per-question in the downloaded results JSON and in the
  question breakdown UI.
- If a follow-up fails to generate (API error), the candidate can retry or skip it —
  a skipped follow-up is excluded from scoring rather than counted wrong.
- Very large cell outputs are truncated at 1,500 characters per cell (marked
  `[output truncated]`); code and markdown are never truncated.
- Quiz generation uses `response_format: json_object` plus Pydantic validation rather
  than a provider-specific structured-output helper, so malformed responses raise a
  clear, user-facing error instead of a crash.
