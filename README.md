# Notebook Comprehension Quiz

A Streamlit webapp for dynamic evaluation of the **employee attrition model** take-home task.
The quiz-taker uploads the Jupyter notebook they submitted; Claude generates a timed
multiple-choice quiz grounded in that specific notebook; answers are scored against the
generated key and Claude writes an evaluator-facing report.

## How it works

1. **Upload** — the candidate uploads their `.ipynb`. The app flattens markdown, code,
   and (truncated) cell outputs into a transcript.
2. **Generate** — Claude Opus 4.8 generates MCQs calibrated against a fixed question
   template (missing-label handling, model comparison, model outputs, metric
   interpretation, explainability), so difficulty is consistent across notebooks and runs.
   Structured outputs guarantee well-formed questions.
3. **Quiz** — a countdown timer (default 5 minutes) runs during the quiz; it auto-submits
   when time expires. Unanswered questions score zero.
4. **Report** — the score is computed against the answer key, and Claude writes a short
   evaluator report (overall assessment, understanding by topic, suggested live follow-up
   questions). Results are downloadable as JSON.

## Setup

```powershell
pip install -r requirements.txt
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # or run `ant auth login`
python -m streamlit run app.py
```

The app opens at http://localhost:8501.

## Configuration

- **Number of questions** (3–10, default 6) and **time limit** (1–15 min, default 5)
  are in the sidebar on the upload screen.
- The question template and difficulty calibration live in `QUESTION_TEMPLATE` in
  `app.py`; the task description shown to the model is in `TASK_CONTEXT`.

## Notes

- Both API calls (generation and grading) share the same system prompt and lead with an
  identical cached notebook block, so the grading call reads the prompt cache written by
  the generation call.
- Very large cell outputs are truncated at 1,500 characters per cell (marked
  `[output truncated]`); code and markdown are never truncated.
