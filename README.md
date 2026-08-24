# Dynamic Evaluation Tools

A multipage Streamlit app with three tools:

1. **Notebook Quiz** — generates a comprehension quiz from an uploaded Jupyter notebook.
2. **Notebook Grader** — grades many uploaded notebooks against a rubric CSV with the
   OpenAI API and exports scores + reasoning to Excel.
3. **Session Timeline** — visualizes a Claude Code session `.jsonl` as an interactive
   swimlane timeline.

```powershell
pip install -r requirements.txt
# create a .env file with:
#   OPENAI_API_KEY = "sk-..."        (needed for Notebook Quiz and Notebook Grader)
#   SUPABASE_URL = "https://xxxx.supabase.co"   (optional -- see Database section below)
#   SUPABASE_KEY = "..."                          (service_role key, not anon)
python -m streamlit run app.py
```

The app opens at http://localhost:8501; use the sidebar/top nav to switch tools.
Page code lives in `app_pages/`; `app.py` is just the `st.navigation` router.
`db.py` (repo root) is the shared Supabase integration both pages can use.

---

## Database (Supabase)

Both the Notebook Quiz and Notebook Grader tools persist to Postgres via
[Supabase](https://supabase.com). **This is optional** — if `SUPABASE_URL`/`SUPABASE_KEY`
aren't set, both tools still work end-to-end (the quiz shows "⚠️ Not saved to database"
but still lets you download JSON; the grader still grades and exports Excel — results
just don't persist across sessions).

The schema creates four tables:
- `quiz_results` — one row per completed quiz (score, per-question breakdown).
- `model_followup_bank` — a growing bank of "benefits and restrictions" questions for
  the quiz's model follow-up, pre-seeded for Logistic Regression / Random Forest / XGBoost.
- `grading_rubric` — a saved grading rubric: the task text plus the uploaded rubric CSV
  stored verbatim, keyed by name.
- `graded_notebooks` — one row per graded notebook (per-item scores + reasoning),
  upserted on (rubric, filename) so re-uploading re-grades rather than duplicating.

### One-time setup

1. Create a free project at [supabase.com](https://supabase.com).
2. Open **SQL Editor** in your project, paste in the contents of `supabase_schema.sql`
   from this repo, and run it. This creates the four tables above.
3. Go to **Project Settings → API** and copy:
   - **Project URL** → `SUPABASE_URL`
   - **`service_role` secret key** (not the `anon` key — the app writes server-side
     and needs to bypass row-level security) → `SUPABASE_KEY`
4. Put those two values in your local `.env` (see above) for local dev, and in your
   deployment's secrets manager when you deploy (see below). **Never commit them** —
   `.streamlit/` and `.env` are already gitignored.

> **Updating an existing database:** `create table if not exists` never alters a table
> that already exists, so a database built from an earlier version of this schema keeps
> its old columns — and a save then fails with `PGRST204 … column not found in the
> schema cache`. The **Migrations** section at the bottom of `supabase_schema.sql`
> handles this: it is idempotent, so just re-run the whole file in the SQL Editor. On a
> fresh database it does nothing; on an older one it adds the missing columns and drops
> the `not null` constraints on retired ones. After migrating `grading_rubric`, re-save
> your rubrics from the Rubric & Task tab so `rubric_csv` is populated.

### Inspecting results

Supabase's own dashboard (**Table Editor**) is a free, ready-made way to browse, filter,
and export any of these tables — no admin page needed.

---

## Notebook Quiz (`app_pages/notebook_quiz.py`)

A Streamlit tool for dynamic evaluation of the **employee attrition model** take-home task.
The quiz-taker uploads the Jupyter notebook they submitted; an OpenAI model generates a quiz
grounded in that specific notebook, mixing fixed/static questions with a couple of questions
personalized to the candidate's own answers; answers are scored against the generated key and
shown in a per-question breakdown. There is no time limit — a stopwatch just tracks how long
the candidate takes.

### How it works

1. **Upload** — the candidate uploads their `.ipynb`. The app flattens markdown, code,
   and (truncated) cell outputs into a transcript.
2. **Generate** — the model generates 8 static questions in one batch call
   (`aggregation`, `train_test_split`, `model_choice`, `io_overlap`, `inputs`,
   `outputs`, `metric_choice`, `metric_interpretation`), following a fixed per-slot
   template (see below) so difficulty and coverage are consistent across notebooks and
   runs. JSON-mode output plus Pydantic validation guarantee well-formed questions. 4
   more fixed, not-LLM-generated questions (`num_employees`, `row_granularity`,
   `label_exists`, `io_overlap_ok`) are interleaved at fixed points in the sequence —
   their correct answers don't depend on the notebook's content. The correct answer and
   explanation are generated/fixed up front along with each question, but never shown
   to the candidate until the final breakdown.
3. **Quiz** — questions are presented one at a time; a stopwatch (not a countdown) shows
   elapsed time, with no limit and no auto-submit. Two questions are personalized live,
   based on the candidate's own preceding answer:
   - **`label_followup`** — only asked if the candidate answers `label_exists` correctly
     ("No" — the dataset's attrition label is actually simulated/synthetic, not real).
     If they answer "Yes" (incorrect) or leave it blank, the follow-up is skipped
     entirely and excluded from scoring (total question count is one less).
   - **`model_followup`** — "What are the benefits and restrictions of using [the
     candidate's chosen model]?" First checked against the `model_followup_bank`
     Supabase table (pre-seeded for Logistic Regression / Random Forest / XGBoost); on
     a miss, the LLM generates 4 plausible options (distractors echo the model's own
     name/keywords so the correct answer isn't obvious by elimination) and the result is
     cached back into the bank via upsert for reuse by future candidates. Options are
     reshuffled on every read so the answer position varies.

   Unanswered questions score zero.
4. **Results** — the score is computed against the answer key and shown as a
   per-question breakdown (your answer, the correct answer, why, and time spent on
   that question). Results — including per-question time spent — are downloadable as
   JSON.

### Question template (fixed slots, in order)

1. **Num employees** *(fixed, not LLM-generated)* — "How many employees are in the
   pq_data dataset?" 100 / 300 / 1000 / 3000; the correct answer is always 300 (pq_data
   always has 300 unique `PersonId` values).
2. **Row granularity** *(fixed, not LLM-generated)* — what a single row in the original
   pq_data represents; the correct answer is always "one employee's metrics per week"
   (pq_data is a weekly Person Query — 35 weekly rows per employee).
3. **Aggregation** — how the notebook aggregated the weekly rows before training (one
   real approach among plausible-but-unused aggregation methods).
4. **Model choice** — fixed 4-option list (Random Forest / XGBoost / Logistic Regression
   / Support Vector Machine); the correct option is whichever the notebook actually used
   (falls back to the closest match if the notebook used a model outside this list).
5. **Model follow-up** *(personalized, DB-backed)* — benefits/restrictions of the
   candidate's chosen model (see above).
6. **Label exists** *(fixed, not LLM-generated)* — "Is there an employee attrition label
   in the dataset?" Yes/No; the correct answer is always "No" (the label is simulated).
7. **Label follow-up** *(personalized, conditional)* — only shown if Q6 was answered
   "No"; asks how the notebook actually addressed the missing/simulated label.
8. **Train/test split** — how the notebook split the data for training vs. evaluation
   (one real approach among plausible-but-unused splitting methods).
9. **IO overlap** — Yes/No: "Are there any variables that were in both the input and
   output of your model?" The correct answer is determined by the LLM reading the
   notebook, not fixed — it's correct if consistent with what the notebook's final model
   actually does.
10. **IO overlap OK** *(fixed, not LLM-generated)* — general-knowledge Yes/No: "Is it
    okay for a variable to be both an input and the output?" The correct answer is
    always "No" (data leakage).
11. **Inputs** — what the final model's actual input features are (one real option among
    plausible-but-wrong input descriptions).
12. **Outputs** — what the final model's actual output/target is (one real option among
    plausible-but-wrong output descriptions).
13. **Evaluation metric** — how the notebook evaluated model performance (one real option
    among plausible unused metric names).
14. **Metric interpretation** — a worked example using the notebook's own reported metric
    value (e.g. "the notebook reports a cross-validated AUC-ROC around 0.53-0.58 — what
    does that mean?"), testing whether the candidate can interpret the number rather than
    just naming the metric.

This is the complete question set — there are no other slots (no EDA-code, training-code,
outline, or production questions).

### Configuration

- Question count varies: 13 questions, plus `label_followup` (making 14) if
  `label_exists` was answered correctly ("No") — there's no time limit or
  question-count setting to configure.
- The per-slot instructions live in `SLOT_INSTRUCTIONS` in `app_pages/notebook_quiz.py`;
  the task description shown to the model is in `TASK_CONTEXT`; the fixed option list
  (`FIXED_MODEL_OPTIONS`) and the fixed (non-LLM) question builders (`num_employees_question`,
  `row_granularity_question`, `label_exists_question`, `io_overlap_ok_question`) are also
  there.
- `MODEL` in `app_pages/notebook_quiz.py` defaults to `gpt-4o`; swap to `gpt-4o-mini`
  for cheaper/faster test iteration.

### Notes

- All API calls share an identical system prompt and lead with the same notebook text,
  so OpenAI's automatic prompt-prefix caching can apply across the batch and follow-up
  calls.
- The two personalized follow-ups (`label_followup`, `model_followup`) are generated
  mid-quiz, based on the candidate's own answer to the preceding question.
  `model_followup` prefers the Supabase bank over calling the LLM; `label_followup` is
  always LLM-generated (there's nothing to bank — it's the same question for everyone)
  but only when triggered.
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

---

## Notebook Grader (`app_pages/notebook_grader.py`)

A Streamlit tool for grading **many** submitted notebooks against a rubric with the
OpenAI API, and exporting scores + reasoning to Excel. Built for grading a batch of
take-home submissions consistently.

### How it works

The page has three tabs:

1. **Rubric & Task** — upload your rubric as a CSV and (optionally) edit the task
   description the notebooks were responding to (defaults to the attrition take-home).
   The rubric CSV is **passed to the model exactly as uploaded** — it is *not* parsed,
   reshaped, or reduced, so every column of grading guidance you wrote (point values,
   full/partial/low-credit descriptions, etc.) reaches the grader intact. Rubric + task
   are saved to Supabase under a name and can be reloaded later.
2. **Grade notebooks** — upload one or many `.ipynb` files. Each notebook is flattened
   to a transcript and graded in its **own** OpenAI API call: the model reads the rubric
   CSV and returns, for every rubric item, a numeric score (0…the item's max, clamped)
   and 1–3 sentences of reasoning citing specific evidence in the notebook. Each result
   is saved per-notebook (upsert on rubric + filename, so re-uploading a filename
   re-grades it). A progress bar tracks the batch; you can upload more notebooks any time.
3. **Results** — a **Summary** table (one row per notebook: per-section totals, grand
   total, max, percent) and a **Details** table (every notebook × rubric item, with the
   reasoning), plus a **Download Excel** button producing a two-sheet workbook
   (`Summary` + `Details`). This tab also has controls to **remove** individual graded
   notebooks or clear all of them for a rubric (deletes from Supabase too).

### Notes

- The rubric structure in the output (sections, item names, max points) comes from the
  model reading your CSV, not from a fixed parse — so the Summary groups by whatever
  section names the model copies back from the CSV. With a clean, consistent CSV this is
  stable; the table builders tolerate minor row differences across notebooks.
- One API call per notebook, all rubric items at once. For a large batch this is the
  main cost driver — switch `MODEL` in `app_pages/notebook_grader.py` to `gpt-4o-mini`
  for cheaper/faster grading.
- Scores are clamped into `[0, max_pts]` server-side, so a model that returns an
  over-max score can't inflate a total.
- Excel export uses `openpyxl` (in `requirements.txt`).

---

## Session Timeline (`app_pages/session_timeline.py`)

Upload a Claude Code session `.jsonl` transcript to see it as a swimlane timeline:
one lane each for **User Prompt**, **Agent Call**, and **Tool Call**, x-axis is
relative session time, and a dotted vertical guide connects the events belonging to
the same "turn." Click any marker to open a popup with that event's content,
session time, and token usage.

### How turns are detected

A **turn** starts at each genuine user prompt — a `type: "user"` record whose message
content is real text, not a tool-result wrapper — and runs until the next one. Within
a turn: each `type: "assistant"` record is an Agent Call event; each `tool_use` block
inside an assistant message is a Tool Call event, labeled with the tool name.
`tool_result` records and any other line types (summary/meta/etc.) aren't plotted on a
lane; tool results are still attached to their Tool Call event for the detail popup.
Malformed or unrecognized lines are skipped and counted, not fatal.

Each user prompt is also classified as **Chat** or **Instruct** via a free, offline
heuristic (question-phrasing → Chat, imperative verbs → Instruct, tool usage in that
turn always wins and forces Instruct) — see `classify_prompt()`.

This parser is a best-effort match to the Claude Code transcript format — if a real
session file doesn't parse the way you expect, the error message and skipped-line
count are the place to start; the parsing logic is in `parse_jsonl()`.

### Notes

- Colors follow a validated categorical palette (blue / aqua / yellow for User
  Prompt / Agent Call / Tool Call) — see `references/palette.md` in the `dataviz`
  skill if you want to re-theme it. Tool Call markers use a distinct symbol per tool
  name (with a legend on the right); User Prompt markers use a distinct symbol per
  Chat/Instruct classification.
- Events that land on nearly the same timestamp within a turn+lane (e.g. several
  tool calls fired from one assistant turn) get a small vertical offset so they
  don't render as one overlapping blob (`_assign_declump_offsets()`).
- Click-to-select uses `st.plotly_chart(..., on_select="rerun")`; every marker's
  `customdata` is that specific event's index, so the popup (`st.dialog`) shows the
  exact event clicked, not just its turn.
- Tokens are only reported where the Anthropic API actually attributes them (on the
  assistant/Agent Call record) — a Tool Call's popup shows the token usage of the
  *agent call that invoked it*, labeled as such, rather than a fabricated number.
- This has been tested against a synthetic sample session, not yet against a real
  Claude Code transcript — try it with one of your own `.jsonl` files and let me know
  if the turn/lane assignment looks off for real data.

---

## Deployment (Streamlit Community Cloud)

Free, and deploys straight from this GitHub repo. These steps involve your own
GitHub/Streamlit accounts, so they're not something that can be scripted for you —
walk through them yourself:

1. Push this repo to GitHub if you haven't already (it's already git-initialized).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in (it offers a
   GitHub-login option — this is the account-authorization step only you can do).
3. Click **New app**, pick this repo/branch, and set the main file path to `app.py`.
4. Before (or right after) the first deploy, open the app's **Settings → Secrets**
   in the Streamlit Cloud dashboard and paste in TOML-formatted secrets, e.g.:
   ```toml
   OPENAI_API_KEY = "sk-..."
   SUPABASE_URL = "https://xxxx.supabase.co"
   SUPABASE_KEY = "..."
   ```
   This is the Cloud equivalent of your local `.env` — `db.py`'s credential loader
   checks `st.secrets` first, so no code changes are needed between local and
   deployed.
5. Deploy. Streamlit Cloud installs `requirements.txt` and runs `app.py` — the
   multipage nav (Notebook Quiz / Session Timeline) works out of the box.

**Tradeoffs to know about:** the free tier is a public-by-default URL (add
`st.secrets`-gated password logic if this needs to be private), apps sleep after
inactivity (a short cold-start delay on the next visit), and there's no custom
domain. If any of that matters — private-only access, always-warm, custom domain —
a small container on Render or Fly.io (~$5–7/mo) trades the one-click simplicity
above for actual infra to manage.
