create table if not exists quiz_results (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  candidate_name text,
  notebook_filename text,
  score integer not null,
  total integer not null,
  elapsed_seconds numeric,
  -- Full per-question breakdown (question text, options, correct/candidate answers,
  -- explanation, time spent) as a JSON array. Stored per-row rather than normalized
  -- into a separate table for now, since the question set is still being iterated on.
  questions jsonb not null
);

create index if not exists quiz_results_created_at_idx on quiz_results (created_at desc);
create index if not exists quiz_results_candidate_name_idx on quiz_results (candidate_name);

-- Row Level Security: enabled by default on new Supabase projects. The app writes
-- using the service_role key (server-side only, bypasses RLS), so no policy is
-- strictly required for the app to function -- but RLS stays ON so the anon/public
-- key (if ever exposed) cannot read or write this table.
alter table quiz_results enable row level security;

-- A small, growing bank of "benefits and restrictions" questions for the model
-- follow-up (asked about whichever model the candidate picked in the model_choice
-- question). Seeded with the three models named in the take-home task; any other
-- model a candidate picks (e.g. Support Vector Machine) gets a question generated
-- on the fly by the LLM and saved here via upsert, so it's reused next time.
create table if not exists model_followup_bank (
  model_name text primary key,
  options jsonb not null,       -- array of exactly 4 option strings
  correct_index integer not null,
  explanation text not null,
  created_at timestamptz not null default now()
);

alter table model_followup_bank enable row level security;

insert into model_followup_bank (model_name, options, correct_index, explanation)
values
  (
    'Logistic Regression',
    '[
      "It provides probabilities and interpretable coefficients, but nonlinear relationships may require additional feature engineering.",
      "It provides probabilities and interpretable coefficients, but only after combining predictions from many fitted trees.",
      "It provides probabilities and nonlinear feature interactions automatically, but cannot show the direction of feature effects.",
      "It provides probabilities and a flexible decision boundary, but usually requires extensive hyperparameter tuning."
    ]'::jsonb,
    0,
    'Logistic Regression gives calibrated probabilities and coefficients whose sign/magnitude are directly interpretable, but it models a linear decision boundary -- capturing nonlinear relationships requires manual feature engineering (interactions, polynomial terms, binning).'
  ),
  (
    'Random Forest',
    '[
      "Its trees are trained sequentially to correct earlier errors, but the learning rate requires careful tuning.",
      "Its trees can capture nonlinear interactions and reduce instability through averaging, but the combined model is harder to explain.",
      "Its trees produce one simple set of decision rules, but the model cannot estimate attrition probabilities.",
      "Its trees work well only after all workplace features have been standardized to the same scale."
    ]'::jsonb,
    1,
    'Random Forest averages many independently-trained trees, which captures nonlinear interactions and reduces the variance/instability of any single tree, at the cost of losing the single-tree interpretability of a simpler model.'
  ),
  (
    'XGBoost',
    '[
      "Its boosting procedure uses regularization and computational optimizations, but tuning and explaining the final model can require additional effort.",
      "Its boosting procedure trains all trees independently, but combining their predictions can require substantial memory.",
      "Its boosting procedure captures nonlinear patterns, but requires every feature to be standardized and cannot handle missing values.",
      "Its boosting procedure includes regularization, so overfitting and probability calibration do not need to be evaluated."
    ]'::jsonb,
    0,
    'XGBoost builds trees sequentially with gradient boosting plus built-in regularization and engineering optimizations (handles missing values, parallelized), often giving strong tabular performance -- but that comes with more hyperparameters to tune and a less directly interpretable model.'
  )
on conflict (model_name) do nothing;


-- ---------------------------------------------------------------------------
-- Notebook grading (Notebook Grader tool)
-- ---------------------------------------------------------------------------

-- A rubric: the task description graded against, plus the rubric CSV stored VERBATIM
-- (exactly as uploaded — the grader hands it to the model unparsed, so every column of
-- guidance is preserved). Keyed by `name` so you can keep multiple rubrics and
-- re-upload/replace one by name.
create table if not exists grading_rubric (
  name text primary key,
  task text not null,
  rubric_csv text not null,   -- the uploaded CSV, verbatim
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table grading_rubric enable row level security;

-- One graded notebook: the flattened transcript plus the model's per-item scores and
-- reasoning. Uniqueness on (rubric_name, notebook_filename) means re-uploading the
-- same notebook under the same rubric re-grades it (upsert) rather than duplicating.
create table if not exists graded_notebooks (
  id uuid primary key default gen_random_uuid(),
  rubric_name text not null references grading_rubric (name) on delete cascade,
  notebook_filename text not null,
  notebook_text text,
  results jsonb not null,   -- [{"section","criterion","max_pts","score","reasoning"}, ...]
  total_score numeric not null,
  max_score numeric not null,
  created_at timestamptz not null default now(),
  unique (rubric_name, notebook_filename)
);

create index if not exists graded_notebooks_rubric_idx
  on graded_notebooks (rubric_name, created_at desc);

alter table graded_notebooks enable row level security;


-- ---------------------------------------------------------------------------
-- Study participants (Participant intake + Admin review)
-- ---------------------------------------------------------------------------
-- Every artifact a participant produces is keyed on subject_id ("P" + 3 digits).
-- One row per participant in every table except participant_files (one row per
-- markdown doc). Child rows cascade-delete with the participant.

create table if not exists participants (
  subject_id text primary key,
  condition text not null,          -- 'slow_planning' | 'slow_iterating' | 'control'
  status text not null default 'in_progress',        -- 'in_progress' (Part 1 unfinished) | 'quiz_done' (Part 1 done, Part 2 owed) | 'complete'
  grading_status text not null default 'pending',     -- 'pending' | 'done' | 'error'
  grading_error text,
  file_manifest jsonb,              -- {expected: [...], received: [...]}
  created_at timestamptz not null default now(),
  submitted_at timestamptz
);

alter table participants enable row level security;

create table if not exists participant_files (
  subject_id text not null references participants (subject_id) on delete cascade,
  doc_type text not null,           -- task_plan | debug_manual | debug_ai | ideate_manual | ideate_ai
  filename text not null,
  content text not null,
  uploaded_at timestamptz not null default now(),
  primary key (subject_id, doc_type)
);

alter table participant_files enable row level security;

create table if not exists participant_notebooks (
  subject_id text primary key references participants (subject_id) on delete cascade,
  filename text not null,
  notebook_text text not null,
  rubric_name text,
  grader_model text,
  results jsonb,                     -- [{"section","criterion","max_pts","score","reasoning"}, ...]
  total_score numeric,
  max_score numeric,
  graded_at timestamptz,
  created_at timestamptz not null default now()
);

alter table participant_notebooks enable row level security;

create table if not exists participant_quiz (
  subject_id text primary key references participants (subject_id) on delete cascade,
  notebook_filename text,
  score integer not null,
  total integer not null,
  elapsed_seconds numeric,
  questions jsonb not null,          -- full per-question breakdown
  generation_warnings jsonb,
  created_at timestamptz not null default now()
);

alter table participant_quiz enable row level security;

create table if not exists participant_logs (
  subject_id text primary key references participants (subject_id) on delete cascade,
  filename text not null,
  raw_jsonl text not null,
  metrics jsonb not null,            -- lib.timeline.compute_log_metrics() output
  parsed_at timestamptz not null default now()
);

alter table participant_logs enable row level security;


-- ---------------------------------------------------------------------------
-- Migrations for databases created from an earlier version of this file
-- ---------------------------------------------------------------------------
-- `create table if not exists` skips a table that already exists, so the blocks
-- above never alter one. Everything below is idempotent and safe to re-run: on a
-- fresh database it is a no-op, on an older one it brings the tables up to date.

-- grading_rubric once stored a parsed rubric (`items` jsonb + `total_points`).
-- The grader now hands the model the uploaded CSV verbatim, so the rubric lives
-- in `rubric_csv` and the two old columns are unused. Without this migration a
-- save fails with: PGRST204 "Could not find the 'rubric_csv' column".
alter table grading_rubric add column if not exists rubric_csv text;

-- Added nullable on purpose: a `not null` column cannot be added to a table that
-- already has rows. Rubrics saved before this migration have it empty -- re-save
-- them from the Rubric & Task tab (upsert by name) to fill it in.

-- The legacy columns must also stop being required, or every insert fails on a
-- not-null violation: the app no longer writes either one. DO blocks because
-- `alter column` has no `if exists` and errors on a fresh database.
do $$
begin
  if exists (select 1 from information_schema.columns
             where table_schema = 'public' and table_name = 'grading_rubric'
                   and column_name = 'items') then
    alter table grading_rubric alter column items drop not null;
  end if;
  if exists (select 1 from information_schema.columns
             where table_schema = 'public' and table_name = 'grading_rubric'
                   and column_name = 'total_points') then
    alter table grading_rubric alter column total_points drop not null;
  end if;
end $$;

-- Optional cleanup, commented out because it permanently deletes the old rubric
-- data. Run it only after re-saving your rubrics; nothing in the app reads these.
-- alter table grading_rubric drop column if exists items,
--                            drop column if exists total_points;

-- PostgREST caches the schema and answers from that cache; this makes the new
-- column visible immediately instead of waiting for its own reload.
notify pgrst, 'reload schema';
