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
| **Participant** | study participants | The whole session in order: surveys, three timed tasks with uploads, the quiz, and an interview hand-off. **No scores are ever shown.** |
| **Admin** | the researcher (password-gated) | per-participant review, cohort statistics, grading controls |

`app.py` is the `st.navigation` router. Page shells live in `app_pages/`; all the
real logic is in `lib/` (`notebook`, `quiz`, `quiz_ui`, `surveys`, `survey_ui`,
`survey_stats`, `tasks`, `grading`, `timeline`, `cohort`, `llm`). `db.py` is the shared Supabase integration. `build_report.py`
is an optional CLI that renders a standalone HTML grading report.

---

## Participant flow (`app_pages/participant.py`)

One continuous run, keyed to a subject ID (`P` + 3 digits, e.g. `P001`). Each
screen the participant reaches is written to `participants.stage`, so re-entering
the same ID resumes **exactly** where they stopped — including a task clock that
was already running.

| # | Stage | What happens |
|---|---|---|
| — | `identify` | Subject ID + the condition assigned in person (typed as text, normalised to `slow_planning` / `slow_iterating` / `control`). |
| 1 | `pre_survey` | The opening survey (`lib/surveys.py`). Every question is required. |
| 2 | `main_task` | The main-task document **for their condition**, then a **40-minute** countdown. |
| 2 | `main_upload` | `Pxxx_task_plan.md` + `Pxxx_notebook.ipynb` (exact names), plus any number of extra files under any name. |
| 3 | `ideate_task` | The ideation document. **Untimed** — the clock counts up and the duration is recorded. **Nothing is uploaded**: the participant pitches their idea aloud, and the researcher uploads the transcript in Admin → Participant → Ideate, where it is graded. |
| 4 | `quiz` | The notebook-comprehension quiz (below). No score is shown. Status → `quiz_done`. |
| 5 | `interview` | Hand-off screen: the participant answers questions verbally with the researcher. Nothing to upload. |
| 6 | `debug_task` | The debugging document, then a **15-minute** countdown. |
| 6 | `debug_upload` | `Pxxx_debug.md`. |
| 7 | `log_upload` | `Pxxx_claude_log.jsonl` — several sessions are fine (`Pxxx_claude_log_2.jsonl`, …). |
| 8 | `post_survey` | The closing survey, then hidden grading runs and status → `complete`. |

**Task instructions are uploaded by the researcher** in Admin → Task instructions:
one main-task document per condition plus the shared ideation and debugging ones.
A task whose document is missing cannot be started.

**Timers are advisory.** The countdown (`lib/tasks.py`) runs client-side in a
same-origin iframe, beeps twice when the allowance runs out, and then keeps
counting into overtime — nothing is locked, and the participant can still finish
and upload. It is anchored to a `started_at` stored in the database, so a refresh
or a dropped connection never hands anyone a fresh allowance; overtime is recorded
for the researcher in Admin → Task timing.

Every upload screen checks names and blocks Submit until they match, so files
always arrive keyed to the right participant.

> **Earlier protocol.** Participants run before this flow have their reflection
> documents split into manual and AI versions (`Pxxx_debug_manual.md`,
> `Pxxx_debug_ai.md`, `Pxxx_ideate_manual.md`, `Pxxx_ideate_ai.md`). Nothing
> writes those any more — new runs collect one `debug` document (and no ideation
> upload: the researcher adds the pitch transcript as the `ideate` document) — but
> Admin still displays and can replace them.

## Comprehension quiz

Claude Opus 5 generates a quiz grounded in the participant's own notebook: 8
LLM-authored questions (aggregation, train/test split, model choice, input/output
overlap, inputs, outputs, evaluation metric, metric interpretation) plus 4 fixed
non-LLM questions and up to 2 live follow-ups personalised to the participant's own
answers. Five slots (aggregation, train/test split, inputs, outputs, evaluation
metric) draw their distractors **verbatim from a hardcoded pool** (`DISTRACTOR_BANK`
in `lib/quiz.py`); the model only picks which pool entry matches the notebook — or
writes the correct option itself, held to the pool's length and detail — so the
correct answer never stands out by being the most specific option. Every LLM-authored
question is then **self-checked** — the model takes the fresh quiz blind and any
question without exactly one defensible answer (or whose correct option is a
specificity giveaway) is repaired and re-checked. Unresolved questions are recorded
as `generation_warnings`. Logic lives in `lib/quiz.py`; the in-app flow and the
admin's read-only breakdown are in `lib/quiz_ui.py`.

## Notebook grading

`lib/grading.py` sends the flattened notebook transcript plus the rubric CSV
(verbatim — never reparsed) to Claude Opus 5, which returns a score + 1–3 sentences
of reasoning per rubric item. The system prompt carries a `cache_control` breakpoint
so batch grading of many notebooks is cheap after the first. Scores are clamped into
`[0, max_pts]`.

## Session-log metrics

`lib/timeline.parse_jsonl()` auto-detects two Claude Code `.jsonl` formats:

- a **session transcript** (`type: user/assistant` + `tool_use` records) — the full
  User Prompt / Agent Call / Tool Call swimlane and all tool metrics.
- the **`history.jsonl`** command-recall file (`display` per line) — **only the
  prompts the participant typed**; it has no assistant responses or tool calls. You
  get a prompt-only timeline (Instruct / Chat / Slash) and prompt-based metrics.

`compute_log_metrics()` returns scalars accordingly: session duration, prompt /
session / tool-call / edit counts, chat-vs-instruct split, token totals, and the
**manipulation-check** metrics — `time_to_first_tool_call_s` /
`median_inter_tool_gap_s` for a transcript, or their history-file proxies
`time_to_first_prompt_s` / `median_inter_prompt_gap_s` (both: first expected highest
under *slow planning*, second under *slow iterating*). The Admin **Grading** tab has
a "Re-parse all logs" button to recompute stored metrics after a parser change.

---

## Admin (`app_pages/admin.py`)

Password gate (`ADMIN_PASSWORD`). Seven tabs:

1. **Overview** — one row per participant (condition, scores, log metrics, grading
   status); filter by condition; download all as CSV.
2. **Participant** — this participant's surveys (with ✅/❌ on the scored knowledge
   items), **task timing** (time spent vs allowance, plus any extra main-task
   files), the task plan / ideation / debugging documents, the graded notebook
   (per-item scores + reasoning + transcript), the quiz breakdown, the **verbal
   assessment** (upload a `.txt` transcript → Claude splits it into timestamped
   question/answer pairs, then a Fact-check grades each answer 0–2 against the
   participant's own notebook), the interactive session timeline with
   click-to-open event detail, and raw file downloads. A "Grade now" button for
   anything still ungraded.
3. **Cohort stats** — outcome and behaviour metrics overall and split by condition
   (box + strip plots), a per-condition n / mean / sd table with a hand-rolled
   one-way ANOVA (F, p, η² — descriptive only), the manipulation checks, and a
   quiz-vs-notebook scatter coloured by condition.
4. **Survey results** — the pre/post surveys across every participant
   (`lib/survey_stats.py`): coverage and timing, pre→post knowledge gain scored
   against `lib.surveys.ANSWER_KEY`, attitude shift, workload, self-assessed vs
   rubric grade, background, free text, and a long-format CSV export.
5. **Task instructions** — upload the markdown each task screen shows: a shared
   main-task **brief** (the task itself, same for every condition) plus one
   main-task **condition instructions** document per condition (slow planning /
   slow iterating / control), shown to the participant on the same page as the
   brief, plus the shared ideation and debugging ones. **Upload all six before
   running anyone** — a task with a missing document cannot be started. Below
   them, the grading/reference material: the **debugging** buggy `.ipynb` and
   the **ideation rubric**. Uploading the buggy notebook also renders it to
   HTML (`lib/notebook.py`'s `notebook_to_html`, stored in
   `task_instructions.notebook_html`); the participant's Debugging task screen
   links to it as an "open in a new tab" page, so they debug the actual
   notebook, not just a description of it. A participant's Debug sub-tab then
   autogrades their write-up (`lib/debug_grading.py`): Opus 5 reads the
   notebook and the write-up, and each bug the participant reported scores +1 if it is
   a real bug and +1 if the fix is valid (so the max is 2 × bugs reported). Their Ideate sub-tab
   grades the pitch transcript against the rubric, read verbatim
   (`lib/ideate_grading.py`); results go to `participant_task_grades`.
6. **Notebook report** — the visual grading report across all graded notebooks.
7. **Grading & rubric** — the active rubric, upload/replace a rubric CSV + task
   text, and "grade all pending participants".

---

## Database (Supabase)

Persistence is via [Supabase](https://supabase.com) Postgres. Without
`SUPABASE_URL` / `SUPABASE_KEY` the app still runs but nothing is stored, so a real
study needs it configured.

Tables (`supabase_schema.sql`):

- `participants` — identity, condition, resume `stage`, submission + grading status
- `participant_files` — the per-task markdown docs (one row per `doc_type`)
- `participant_extra_files` — anything extra attached to the main task (binaries
  are base64'd, which `encoding` records)
- `participant_notebooks` — the `.ipynb` transcript + its rubric grading
- `participant_task_grades` — the autograde of a participant's debugging write-up
  or ideation pitch transcript, one row per (participant, task). The grading material
  is `task_instructions` rows: `debug_notebook`, `ideate_rubric`.
- `participant_quiz` — the comprehension-quiz result + per-question breakdown
- `participant_surveys` — the pre/post survey responses + per-item timing
- `participant_task_timings` — when each timed task was started and finished
- `task_instructions` — the admin-authored markdown for each task screen, plus
  `notebook_html` on the `debug_notebook` row (the buggy notebook rendered to
  HTML for the participant to view)
- `participant_logs` — the raw session `.jsonl` + derived metrics
- `participant_transcripts` — the admin-uploaded verbal-assessment `.txt` + its
  parsed timestamped Q/A pairs
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
