# Dynamic Evaluation Study Platform

A multipage Streamlit app for running the delegation-conditions study: participants
submit their work under a subject ID, everything is graded and analysed behind the
scenes, and a researcher reviews it all behind an admin gate.

```powershell
pip install -r requirements.txt
# create a .env file with:
#   ANTHROPIC_API_KEY = "sk-ant-..."   (quiz generation + notebook grading; Claude Opus 5)
#   ADMIN_PASSWORD    = "..."           (gates the Admin page)
#   SUPABASE_URL      = "https://xxxx.supabase.co"
#   SUPABASE_KEY      = "..."           (service_role key, not anon)
#   ACTIVE_RUBRIC_NAME = "attrition_v1" (optional; which stored rubric grading uses)
python -m streamlit run app.py
```

The app opens at http://localhost:8501 with two pages:

| Page | Who | What |
|---|---|---|
| **Participant** | study participants | Part 1: subject ID + condition, upload the notebook, take the quiz. Part 2 (later, same ID): upload the reflection files + session log. **No scores are ever shown.** |
| **Admin** | the researcher (password-gated) | per-participant review, cohort statistics, grading controls |

`app.py` is the `st.navigation` router. Page shells live in `app_pages/`; all the
real logic is in `lib/` (`notebook`, `quiz`, `quiz_ui`, `grading`, `timeline`,
`cohort`, `llm`). `db.py` is the shared Supabase integration. `build_report.py`
is an optional CLI that renders a standalone HTML grading report.

---

## Participant flow (`app_pages/participant.py`)

A **two-part** flow, both parts keyed to the same subject ID (`P` + 3 digits, e.g.
`P001`). The page routes on the participant's stored status when they enter their ID.

**Part 1 — right after the coding task:**

1. **Identify** — enter the subject ID and the condition assigned in person (typed
   as text, normalised to `slow_planning` / `slow_iterating` / `control`).
2. **Notebook** — upload only `Pxxx_notebook.ipynb` (exact name required). The
   transcript is stored.
3. **Quiz** — the notebook-comprehension quiz (see below) runs in-app. The
   participant never sees a score or breakdown.
4. **Finalize** — the notebook is graded synchronously against the active rubric
   behind a neutral spinner; grading failures are recorded for the admin, never
   shown. Status → `quiz_done`. The participant can go straight to Part 2 or leave.

**Part 2 — any time later:**

Re-enter the same subject ID → the page detects `quiz_done` and asks for the
remaining **6 files**, each named `{subject_id}_<name>` exactly:

- `Pxxx_task_plan.md`
- `Pxxx_debug_manual.md`, `Pxxx_debug_ai.md`
- `Pxxx_ideate_manual.md`, `Pxxx_ideate_ai.md`
- `Pxxx_claude_log.jsonl`

A file check lists ✅ found / ❌ missing / 🚫 unexpected; **Submit is blocked until
exactly those 6 correctly-named files are selected.** The notebook is never
re-uploaded. The five docs plus the raw session log (and its derived behaviour
metrics) are saved, and status → `complete`.

## Comprehension quiz

Claude Opus 5 generates a quiz grounded in the participant's own notebook: 8
LLM-authored questions (aggregation, train/test split, model choice, input/output
overlap, inputs, outputs, evaluation metric, metric interpretation) plus 4 fixed
non-LLM questions and up to 2 live follow-ups personalised to the participant's own
answers. Every LLM-authored question is **self-checked** — the model takes the fresh
quiz blind and any question without exactly one defensible answer is repaired and
re-checked. Unresolved questions are recorded as `generation_warnings`. Logic lives
in `lib/quiz.py`; the in-app flow and the admin's read-only breakdown are in
`lib/quiz_ui.py`.

## Notebook grading

`lib/grading.py` sends the flattened notebook transcript plus the rubric CSV
(verbatim — never reparsed) to Claude Opus 5, which returns a score + 1–3 sentences
of reasoning per rubric item. The system prompt carries a `cache_control` breakpoint
so batch grading of many notebooks is cheap after the first. Scores are clamped into
`[0, max_pts]`.

## Session-log metrics

`lib/timeline.compute_log_metrics()` reduces a parsed Claude Code `.jsonl` into
scalars: session duration, turn / prompt / tool-call / edit counts, tool breakdown,
chat-vs-instruct split, token totals, and two **manipulation-check** metrics —
`time_to_first_tool_call_s` (expected highest under *slow planning*) and
`median_inter_tool_gap_s` (expected highest under *slow iterating*).

---

## Admin (`app_pages/admin.py`)

Password gate (`ADMIN_PASSWORD`). Four tabs:

1. **Overview** — one row per participant (condition, scores, log metrics, grading
   status); filter by condition; download all as CSV.
2. **Participant** — the five markdown docs (manual vs AI side by side), the graded
   notebook (per-item scores + reasoning + transcript), the quiz breakdown, the
   interactive session timeline with click-to-open event detail, and raw file
   downloads. A "Grade now" button for anything still ungraded.
3. **Cohort stats** — outcome and behaviour metrics overall and split by condition
   (box + strip plots), a per-condition n / mean / sd table with a hand-rolled
   one-way ANOVA (F, p, η² — descriptive only), the manipulation checks, and a
   quiz-vs-notebook scatter coloured by condition.
4. **Grading & rubric** — the active rubric, upload/replace a rubric CSV + task
   text, and "grade all pending participants".

---

## Database (Supabase)

Persistence is via [Supabase](https://supabase.com) Postgres. Without
`SUPABASE_URL` / `SUPABASE_KEY` the app still runs but nothing is stored, so a real
study needs it configured.

Tables (`supabase_schema.sql`):

- `participants` — identity, condition, submission + grading status
- `participant_files` — the 5 markdown docs (one row per `doc_type`)
- `participant_notebooks` — the `.ipynb` transcript + its rubric grading
- `participant_quiz` — the comprehension-quiz result + per-question breakdown
- `participant_logs` — the raw session `.jsonl` + derived metrics
- `grading_rubric` — a task description + rubric CSV stored verbatim, keyed by name
- `model_followup_bank` — the growing bank of quiz model-follow-up questions

### One-time setup

1. Create a project at [supabase.com](https://supabase.com).
2. Open **SQL Editor**, paste in `supabase_schema.sql`, and run it. It is
   idempotent — safe to re-run on an existing database to add the participant
   tables.
3. **Project Settings → API**: copy the **Project URL** → `SUPABASE_URL` and the
   **`service_role` secret key** → `SUPABASE_KEY`.
4. Put those (and `ANTHROPIC_API_KEY`, `ADMIN_PASSWORD`) in `.env` locally and in
   your deployment's secrets manager. Never commit them.

Supabase's **Table Editor** is a ready-made way to browse and export any table.

---

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io) → **New app**, pick the
   repo/branch, main file `app.py`.
3. **Settings → Secrets**: paste TOML-formatted secrets (`ANTHROPIC_API_KEY`,
   `ADMIN_PASSWORD`, `SUPABASE_URL`, `SUPABASE_KEY`, optionally
   `ACTIVE_RUBRIC_NAME`) as **top-level** keys, not nested under a `[section]`.
4. Deploy.

The free tier is public-by-default — the Admin page is only as private as
`ADMIN_PASSWORD`. For a truly private deployment use a small container on Render or
Fly.io.
