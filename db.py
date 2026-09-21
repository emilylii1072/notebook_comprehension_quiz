"""Shared Supabase (Postgres) integration for the Dynamic Evaluation Tools app.

Credentials are read from `st.secrets` first (Streamlit Community Cloud's secrets
manager) and fall back to environment variables (populated from a local `.env` via
python-dotenv, for local dev). If neither is configured, `get_supabase_client()`
returns None and callers should degrade gracefully -- the app must keep working
without a database attached.
"""

import base64
import os
import random
from datetime import datetime, timezone

import streamlit as st
from supabase import Client, create_client

# The three delegation conditions a participant can be assigned to.
CONDITIONS = ("slow_planning", "slow_iterating", "control")

# doc_type values stored in participant_files (the five markdown documents).
PARTICIPANT_DOC_TYPES = (
    "task_plan", "debug_manual", "debug_ai", "ideate_manual", "ideate_ai",
)


def _get_secret(key: str) -> str | None:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass  # no secrets.toml at all -- st.secrets raises rather than returning {}
    return os.environ.get(key)


def get_secret(key: str, default: str | None = None) -> str | None:
    """Public accessor for a config value (st.secrets first, then env/.env)."""
    value = _get_secret(key)
    return value if value is not None else default


@st.cache_resource
def get_supabase_client() -> Client | None:
    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


# ---------------------------------------------------------------------------
# Diagnostics -- "why isn't it saving?" without ever exposing a credential
# ---------------------------------------------------------------------------

def _key_role(key: str) -> str:
    """Best-effort role of a Supabase key. The app needs the service_role key: it
    writes server-side and must bypass row-level security."""
    if key.startswith("sb_secret_"):
        return "service_role (new-style sb_secret_ key)"
    if key.startswith("sb_publishable_"):
        return "PUBLISHABLE -- wrong key, writes will be blocked by RLS"
    parts = key.split(".")
    if len(parts) == 3:  # legacy JWT -- read the unverified payload for its role
        import base64
        import json
        try:
            pad = parts[1] + "=" * (-len(parts[1]) % 4)
            role = json.loads(base64.urlsafe_b64decode(pad)).get("role", "")
        except Exception:
            return "unknown (could not decode)"
        if role == "service_role":
            return "service_role"
        if role:
            return f"{role.upper()} -- wrong key, writes will be blocked by RLS"
    return "unknown format"


def supabase_status() -> dict:
    """What the running process can actually see, for troubleshooting a deployment.

    Returns only presence/shape facts -- never a secret value. `top_level_secrets`
    is the key names st.secrets exposes: if the credentials were pasted under a
    TOML section header they show up as the section name instead of SUPABASE_URL,
    which is the usual reason a "configured" app still reports it isn't."""
    info: dict = {"secrets_error": None, "top_level_secrets": []}
    try:
        info["top_level_secrets"] = sorted(st.secrets.keys())
    except Exception as e:
        info["secrets_error"] = str(e)  # no secrets.toml at all (normal locally)

    for name in ("SUPABASE_URL", "SUPABASE_KEY", "ANTHROPIC_API_KEY"):
        in_secrets = name in info["top_level_secrets"]
        value = _get_secret(name)
        info[name] = {
            "found": bool(value),
            "source": "st.secrets" if in_secrets else ("env/.env" if value else "-"),
            "length": len(value) if value else 0,
            "has_whitespace": bool(value) and value != value.strip(),
        }
    url = _get_secret("SUPABASE_URL")
    if url:
        info["SUPABASE_URL"]["looks_like_url"] = url.strip().startswith("https://")
    key = _get_secret("SUPABASE_KEY")
    if key:
        info["SUPABASE_KEY"]["role"] = _key_role(key.strip())

    info["client_created"] = get_supabase_client() is not None
    return info


def test_connection() -> tuple[bool, str]:
    """Round-trip a real read against the database. Returns (ok, message)."""
    client = get_supabase_client()
    if client is None:
        return False, "No client -- SUPABASE_URL/SUPABASE_KEY not visible to the app."
    try:
        resp = client.table("grading_rubric").select("name", count="exact").limit(1).execute()
        return True, f"Connected. grading_rubric has {resp.count} row(s)."
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def save_quiz_result(payload: dict) -> tuple[bool, str | None]:
    """Insert one row into the `quiz_results` table.

    Returns (success, error_message). Never raises -- a database outage should not
    block the candidate from seeing their results or downloading the JSON.
    """
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured (SUPABASE_URL/SUPABASE_KEY not set)."
    try:
        client.table("quiz_results").insert(payload).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def get_model_followup(model_name: str) -> dict | None:
    """Look up a stored benefits/restrictions question for this model name.

    Returns {"options": [4 strings], "correct_index": int, "explanation": str} with
    the options reshuffled (correct_index adjusted to match) on every read, so
    candidates who hit the same bank entry don't see the answer in a fixed position.
    Returns None if unconfigured, unreachable, or this model isn't banked yet.
    """
    client = get_supabase_client()
    if client is None:
        return None
    try:
        resp = (
            client.table("model_followup_bank")
            .select("options,correct_index,explanation")
            .eq("model_name", model_name)
            .limit(1)
            .execute()
        )
    except Exception:
        return None
    rows = resp.data or []
    if not rows:
        return None

    row = rows[0]
    options = list(row["options"])
    correct_text = options[row["correct_index"]]
    order = list(range(len(options)))
    random.shuffle(order)
    shuffled = [options[i] for i in order]
    return {
        "options": shuffled,
        "correct_index": shuffled.index(correct_text),
        "explanation": row["explanation"],
    }


def save_model_followup(
    model_name: str, options: list[str], correct_index: int, explanation: str
) -> None:
    """Cache a freshly LLM-generated model follow-up question for reuse. Best-effort
    -- a save failure should never block showing the question to the current
    candidate, so this never raises."""
    client = get_supabase_client()
    if client is None:
        return
    try:
        client.table("model_followup_bank").upsert(
            {
                "model_name": model_name,
                "options": options,
                "correct_index": correct_index,
                "explanation": explanation,
            },
            on_conflict="model_name",
        ).execute()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Notebook grading
# ---------------------------------------------------------------------------

def save_rubric(
    name: str, task: str, rubric_csv: str, grading_instructions: str | None = None
) -> tuple[bool, str | None]:
    """Upsert a rubric (task + the raw uploaded CSV + optional grading-instructions
    override) by name. Returns (success, error). `grading_instructions` is the
    "how to grade" text handed to the model in place of
    lib.grading.DEFAULT_GRADING_INSTRUCTIONS -- None/'' means use the default."""
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured (SUPABASE_URL/SUPABASE_KEY not set)."
    try:
        client.table("grading_rubric").upsert(
            {
                "name": name,
                "task": task,
                "rubric_csv": rubric_csv,
                "grading_instructions": grading_instructions or None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="name",
        ).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def get_rubric(name: str) -> dict | None:
    """Fetch a stored rubric by name, or None if unconfigured/missing."""
    client = get_supabase_client()
    if client is None:
        return None
    try:
        resp = (
            client.table("grading_rubric")
            .select("name,task,rubric_csv,grading_instructions,updated_at")
            .eq("name", name)
            .limit(1)
            .execute()
        )
    except Exception:
        return None
    rows = resp.data or []
    return rows[0] if rows else None


def list_rubrics() -> list[str]:
    """Names of all stored rubrics (newest first). Empty list if unconfigured."""
    client = get_supabase_client()
    if client is None:
        return []
    try:
        resp = (
            client.table("grading_rubric")
            .select("name,created_at")
            .order("created_at", desc=True)
            .execute()
        )
    except Exception:
        return []
    return [r["name"] for r in (resp.data or [])]


def save_graded_notebook(payload: dict) -> tuple[bool, str | None]:
    """Upsert one graded notebook (re-grades on same rubric_name+filename).
    Returns (success, error). Never raises."""
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured (SUPABASE_URL/SUPABASE_KEY not set)."
    try:
        client.table("graded_notebooks").upsert(
            payload, on_conflict="rubric_name,notebook_filename"
        ).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def delete_graded_notebook(rubric_name: str, notebook_filename: str) -> tuple[bool, str | None]:
    """Delete one graded notebook (by rubric + filename). Returns (success, error).
    Success is also returned when there's no DB configured, so the caller can still
    drop it from the in-session view."""
    client = get_supabase_client()
    if client is None:
        return True, None
    try:
        (
            client.table("graded_notebooks")
            .delete()
            .eq("rubric_name", rubric_name)
            .eq("notebook_filename", notebook_filename)
            .execute()
        )
        return True, None
    except Exception as e:
        return False, str(e)


def delete_all_graded_notebooks(rubric_name: str) -> tuple[bool, str | None]:
    """Delete every graded notebook for a rubric. Returns (success, error)."""
    client = get_supabase_client()
    if client is None:
        return True, None
    try:
        client.table("graded_notebooks").delete().eq("rubric_name", rubric_name).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def get_graded_notebooks(rubric_name: str, include_text: bool = False) -> list[dict]:
    """All graded notebooks for a rubric (newest first). Empty if unconfigured.

    `include_text` also fetches the stored notebook transcript -- off by default
    because the app's tables never show it and it dominates the payload size;
    build_report.py turns it on to embed the transcript in its review pages."""
    client = get_supabase_client()
    if client is None:
        return []
    columns = "notebook_filename,results,total_score,max_score,created_at"
    if include_text:
        columns += ",notebook_text"
    try:
        resp = (
            client.table("graded_notebooks")
            .select(columns)
            .eq("rubric_name", rubric_name)
            .order("created_at", desc=True)
            .execute()
        )
    except Exception:
        return []
    return resp.data or []


# ---------------------------------------------------------------------------
# Study participants
# ---------------------------------------------------------------------------
# Every artifact a participant produces is keyed on their `subject_id` (format
# "P" + 3 digits, e.g. P001). Tables:
#   participants          -- identity, condition, submission/grading status
#   participant_files     -- the 5 markdown docs (one row per doc_type)
#   participant_notebooks -- the .ipynb transcript + its rubric grading
#   participant_quiz      -- the in-app comprehension quiz result
#   participant_logs      -- the Claude Code session .jsonl + derived metrics
# All the helpers below follow the module contract: never raise, return
# (ok, error_message) or None/[], and degrade cleanly when no client is configured.


def participant_exists(subject_id: str) -> bool:
    client = get_supabase_client()
    if client is None:
        return False
    try:
        resp = (
            client.table("participants")
            .select("subject_id")
            .eq("subject_id", subject_id)
            .limit(1)
            .execute()
        )
        return bool(resp.data)
    except Exception:
        return False


def upsert_participant(subject_id: str, condition: str) -> tuple[bool, str | None]:
    """Create or reset a participant row. Resetting (a re-submission) also wipes the
    child rows so a partial re-run can't leave stale files/grades behind."""
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured (SUPABASE_URL/SUPABASE_KEY not set)."
    try:
        for table in ("participant_files", "participant_notebooks",
                      "participant_quiz", "participant_logs",
                      "participant_turn_annotations", "participant_surveys",
                      "participant_task_timings", "participant_extra_files"):
            client.table(table).delete().eq("subject_id", subject_id).execute()
        client.table("participants").upsert(
            {
                "subject_id": subject_id,
                "condition": condition,
                "status": "in_progress",
                "grading_status": "pending",
                "grading_error": None,
                "submitted_at": None,
                "file_manifest": None,
                "stage": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="subject_id",
        ).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def save_participant_file(
    subject_id: str, doc_type: str, filename: str, content: str
) -> tuple[bool, str | None]:
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured."
    try:
        client.table("participant_files").upsert(
            {
                "subject_id": subject_id,
                "doc_type": doc_type,
                "filename": filename,
                "content": content,
            },
            on_conflict="subject_id,doc_type",
        ).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def delete_participant_file(subject_id: str, doc_type: str) -> tuple[bool, str | None]:
    """Remove one of a participant's reflection docs (by doc_type)."""
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured."
    try:
        client.table("participant_files").delete().eq(
            "subject_id", subject_id
        ).eq("doc_type", doc_type).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def save_participant_notebook(
    subject_id: str, filename: str, notebook_text: str
) -> tuple[bool, str | None]:
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured."
    try:
        client.table("participant_notebooks").upsert(
            {
                "subject_id": subject_id,
                "filename": filename,
                "notebook_text": notebook_text,
            },
            on_conflict="subject_id",
        ).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def delete_participant_notebook(subject_id: str) -> tuple[bool, str | None]:
    """Remove the participant's notebook (grade included, same row). Also clears
    verbal-assessment fact-checks, since they were checked against this notebook."""
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured."
    try:
        client.table("participant_notebooks").delete().eq("subject_id", subject_id).execute()
    except Exception as e:
        return False, str(e)
    delete_turn_annotations(subject_id, "transcript")
    return True, None


def update_participant_grading(
    subject_id: str,
    rubric_name: str,
    grader_model: str,
    results: list[dict],
    total_score: float,
    max_score: float,
) -> tuple[bool, str | None]:
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured."
    try:
        client.table("participant_notebooks").update(
            {
                "rubric_name": rubric_name,
                "grader_model": grader_model,
                "results": results,
                "total_score": total_score,
                "max_score": max_score,
                "graded_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("subject_id", subject_id).execute()
        client.table("participants").update(
            {"grading_status": "done", "grading_error": None}
        ).eq("subject_id", subject_id).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def save_task_grade(
    subject_id: str, task_key: str, results: list[dict], total_score: float,
    max_score: float, notes: str, grader_model: str,
) -> tuple[bool, str | None]:
    """Upsert the autograde of one task's admin-graded material -- the debugging
    write-up ('debug') or the ideation pitch transcript ('ideate'). One per
    (participant, task); re-grading replaces it."""
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured."
    try:
        client.table("participant_task_grades").upsert(
            {
                "subject_id": subject_id,
                "task_key": task_key,
                "results": results,
                "total_score": total_score,
                "max_score": max_score,
                "notes": notes or None,
                "grader_model": grader_model,
                "graded_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="subject_id,task_key",
        ).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def get_task_grade(subject_id: str, task_key: str) -> dict | None:
    """The stored autograde for one task, or None -- also when the
    participant_task_grades table doesn't exist yet."""
    client = get_supabase_client()
    if client is None:
        return None
    try:
        rows = (
            client.table("participant_task_grades").select("*")
            .eq("subject_id", subject_id).eq("task_key", task_key)
            .limit(1).execute().data or []
        )
    except Exception:
        return None
    return rows[0] if rows else None


def set_participant_grading_error(subject_id: str, message: str) -> None:
    client = get_supabase_client()
    if client is None:
        return
    try:
        client.table("participants").update(
            {"grading_status": "error", "grading_error": message[:2000]}
        ).eq("subject_id", subject_id).execute()
    except Exception:
        pass


def save_participant_quiz(subject_id: str, payload: dict) -> tuple[bool, str | None]:
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured."
    try:
        client.table("participant_quiz").upsert(
            {
                "subject_id": subject_id,
                "notebook_filename": payload.get("notebook_filename"),
                "score": payload["score"],
                "total": payload["total"],
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "questions": payload["questions"],
                "generation_warnings": payload.get("generation_warnings") or [],
            },
            on_conflict="subject_id",
        ).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def save_participant_survey(
    subject_id: str, survey_type: str, responses: list[dict], elapsed_seconds: float | None
) -> tuple[bool, str | None]:
    """Upsert one participant's pre- or post-survey responses (survey_type is
    'pre' or 'post'). `responses` is the full per-item list -- see
    lib.surveys.SurveyItem / lib.survey_ui for the shape."""
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured."
    try:
        client.table("participant_surveys").upsert(
            {
                "subject_id": subject_id,
                "survey_type": survey_type,
                "responses": responses,
                "elapsed_seconds": elapsed_seconds,
            },
            on_conflict="subject_id,survey_type",
        ).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def get_participant_survey(subject_id: str, survey_type: str) -> dict | None:
    """Fetch one participant's pre- or post-survey record, or None if they
    haven't taken it yet. Used both to resume-skip a completed survey and by
    Admin to display it."""
    client = get_supabase_client()
    if client is None:
        return None
    try:
        resp = (
            client.table("participant_surveys")
            .select("*")
            .eq("subject_id", subject_id).eq("survey_type", survey_type)
            .limit(1)
            .execute()
        )
    except Exception:
        return None
    rows = resp.data or []
    return rows[0] if rows else None


def save_participant_log(
    subject_id: str, filename: str, raw_jsonl: str, metrics: dict
) -> tuple[bool, str | None]:
    """Add or replace one of a participant's (possibly several) log files, keyed
    by (subject_id, filename) — a new filename adds a log, the same filename
    replaces just that one. If the raw content actually changed, any turn
    annotations already saved against this exact file are cleared, since new
    content makes their turn_index-keyed tags meaningless (same lesson as a
    stale rubric/taxonomy: don't let old annotations silently linger against
    replaced content). A pure re-parse (metrics recomputed, content unchanged —
    e.g. after a parser fix) leaves existing annotations alone."""
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured."
    try:
        prior = (
            client.table("participant_logs").select("raw_jsonl")
            .eq("subject_id", subject_id).eq("filename", filename)
            .limit(1).execute().data or []
        )
        content_changed = not prior or prior[0].get("raw_jsonl") != raw_jsonl
        client.table("participant_logs").upsert(
            {
                "subject_id": subject_id,
                "filename": filename,
                "raw_jsonl": raw_jsonl,
                "metrics": metrics,
            },
            on_conflict="subject_id,filename",
        ).execute()
    except Exception as e:
        return False, str(e)
    if content_changed:
        delete_turn_annotations(subject_id, "log", filename)
    return True, None


def delete_participant_log(subject_id: str, filename: str) -> tuple[bool, str | None]:
    """Remove one of a participant's log files (by filename) and its turn
    annotations, leaving any other logs on file untouched."""
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured."
    try:
        client.table("participant_logs").delete().eq(
            "subject_id", subject_id
        ).eq("filename", filename).execute()
    except Exception as e:
        return False, str(e)
    delete_turn_annotations(subject_id, "log", filename)
    return True, None


def mark_participant_complete(
    subject_id: str, file_manifest: dict
) -> tuple[bool, str | None]:
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured."
    try:
        client.table("participants").update(
            {
                "status": "complete",
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "file_manifest": file_manifest,
            }
        ).eq("subject_id", subject_id).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def get_participant(subject_id: str) -> dict | None:
    client = get_supabase_client()
    if client is None:
        return None
    try:
        resp = (
            client.table("participants")
            .select("*")
            .eq("subject_id", subject_id)
            .limit(1)
            .execute()
        )
    except Exception:
        return None
    rows = resp.data or []
    return rows[0] if rows else None


def save_participant_transcript(
    subject_id: str, filename: str, raw_text: str, parsed: list[dict] | None
) -> tuple[bool, str | None]:
    """Upsert the admin-uploaded verbal-assessment transcript (one per participant)."""
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured."
    try:
        client.table("participant_transcripts").upsert(
            {
                "subject_id": subject_id,
                "filename": filename,
                "raw_text": raw_text,
                "parsed": parsed,
                "parsed_at": datetime.now(timezone.utc).isoformat() if parsed is not None else None,
            },
            on_conflict="subject_id",
        ).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def get_participant_bundle(subject_id: str) -> dict:
    """Everything stored for one participant, for the admin detail view. Missing
    pieces come back as None / [] rather than raising."""
    client = get_supabase_client()
    empty = {
        "participant": None, "files": [], "notebook": None, "quiz": None,
        "logs": [], "transcript": None, "pre_survey": None, "post_survey": None,
        "task_timings": {}, "extra_files": [],
    }
    if client is None:
        return empty
    out = dict(empty)
    try:
        out["participant"] = get_participant(subject_id)
        out["files"] = (
            client.table("participant_files").select("*")
            .eq("subject_id", subject_id).execute().data or []
        )
        nb = (
            client.table("participant_notebooks").select("*")
            .eq("subject_id", subject_id).limit(1).execute().data or []
        )
        out["notebook"] = nb[0] if nb else None
        qz = (
            client.table("participant_quiz").select("*")
            .eq("subject_id", subject_id).limit(1).execute().data or []
        )
        out["quiz"] = qz[0] if qz else None
        out["logs"] = sorted(
            client.table("participant_logs").select("*")
            .eq("subject_id", subject_id).execute().data or [],
            key=lambda r: r.get("parsed_at") or "",
        )
        tr = (
            client.table("participant_transcripts").select("*")
            .eq("subject_id", subject_id).limit(1).execute().data or []
        )
        out["transcript"] = tr[0] if tr else None
        out["pre_survey"] = get_participant_survey(subject_id, "pre")
        out["post_survey"] = get_participant_survey(subject_id, "post")
        out["task_timings"] = list_participant_task_timings(subject_id)
        out["extra_files"] = list_participant_extra_files(subject_id)
    except Exception:
        pass
    return out


def list_participant_summaries() -> list[dict]:
    """One flattened row per participant for the admin overview table and the
    cohort statistics. Joins are done here in Python (small N)."""
    client = get_supabase_client()
    if client is None:
        return []
    try:
        parts = (
            client.table("participants").select("*")
            .order("created_at", desc=True).execute().data or []
        )
        notebooks = {
            r["subject_id"]: r
            for r in (client.table("participant_notebooks")
                      .select("subject_id,total_score,max_score,graded_at").execute().data or [])
        }
        quizzes = {
            r["subject_id"]: r
            for r in (client.table("participant_quiz")
                      .select("subject_id,score,total,elapsed_seconds").execute().data or [])
        }
        log_rows: dict[str, list[dict]] = {}
        for r in (client.table("participant_logs")
                  .select("subject_id,raw_jsonl").execute().data or []):
            log_rows.setdefault(r["subject_id"], []).append(r)
        transcript_ids = {
            r["subject_id"]
            for r in (client.table("participant_transcripts")
                      .select("subject_id").execute().data or [])
        }
    except Exception:
        return []

    # A participant can have several log files now; combine them the same way
    # the admin's Session timeline does (one merged event stream, one metrics
    # pass) rather than reading back each log's own stored `metrics` column --
    # per-log scalars like a median gap can't be validly combined after the
    # fact. This is an intentional exception to db.py's usual no-lib-imports
    # rule: it's the same "recompute live" approach already used for a single
    # log (see the Session timeline sub-tab), just extended to several.
    from lib.timeline import compute_log_metrics, merge_parsed, parse_jsonl

    rows = []
    for p in parts:
        sid = p["subject_id"]
        nb = notebooks.get(sid) or {}
        qz = quizzes.get(sid) or {}
        my_logs = log_rows.get(sid) or []
        metrics = (
            compute_log_metrics(merge_parsed([parse_jsonl(r["raw_jsonl"]) for r in my_logs]))
            if my_logs else {}
        )
        total, mx = nb.get("total_score"), nb.get("max_score")
        rows.append(
            {
                "subject_id": sid,
                "condition": p.get("condition"),
                "status": p.get("status"),
                "grading_status": p.get("grading_status"),
                "submitted_at": p.get("submitted_at"),
                "notebook_total": total,
                "notebook_max": mx,
                "notebook_pct": (100 * total / mx) if (total is not None and mx) else None,
                "quiz_score": qz.get("score"),
                "quiz_total": qz.get("total"),
                "quiz_pct": (100 * qz["score"] / qz["total"])
                if (qz.get("score") is not None and qz.get("total")) else None,
                "log_format": metrics.get("format"),
                "n_logs": len(my_logs),
                "session_duration_s": metrics.get("session_duration_s"),
                "n_tool_calls": metrics.get("n_tool_calls"),
                "n_edits": metrics.get("n_edits"),
                "n_prompts": metrics.get("n_substantive_prompts"),
                "n_sessions": metrics.get("n_sessions"),
                "time_to_first_tool_call_s": metrics.get("time_to_first_tool_call_s"),
                "time_to_first_prompt_s": metrics.get("time_to_first_prompt_s"),
                "median_inter_tool_gap_s": metrics.get("median_inter_tool_gap_s"),
                "median_inter_prompt_gap_s": metrics.get("median_inter_prompt_gap_s"),
                "chat_count": metrics.get("chat_count"),
                "instruct_count": metrics.get("instruct_count"),
                "has_transcript": sid in transcript_ids,
            }
        )
    return rows


def list_participant_logs_raw() -> list[dict]:
    """[{subject_id, filename, raw_jsonl}] — for re-parsing stored logs after a
    parser change."""
    client = get_supabase_client()
    if client is None:
        return []
    try:
        return (
            client.table("participant_logs")
            .select("subject_id,filename,raw_jsonl").execute().data or []
        )
    except Exception:
        return []


def list_participant_transcripts_raw() -> list[dict]:
    """[{subject_id, filename, parsed, notebook_text}] — for bulk fact-checking
    verbal-assessment transcripts against each participant's own notebook.
    `parsed` is [{"timestamp","question","answer"}, ...] or None if the transcript
    was uploaded but never parsed. `notebook_text` is None if no notebook is on
    file for that participant yet (nothing to check the answer against)."""
    client = get_supabase_client()
    if client is None:
        return []
    try:
        rows = (
            client.table("participant_transcripts")
            .select("subject_id,filename,parsed").execute().data or []
        )
        notebooks = {
            r["subject_id"]: r.get("notebook_text")
            for r in (client.table("participant_notebooks")
                      .select("subject_id,notebook_text").execute().data or [])
        }
    except Exception:
        return []
    for r in rows:
        r["notebook_text"] = notebooks.get(r["subject_id"])
    return rows


def delete_turn_annotations(
    subject_id: str, source: str, log_filename: str = ""
) -> tuple[bool, str | None]:
    """Remove turn annotations for (subject_id, source[, log_filename]) — use
    after the underlying file (log/transcript) or its ground truth (notebook)
    was replaced, since old turn_index-keyed annotations no longer correspond
    to real content. `log_filename` is ignored (matches the default '') for
    source='transcript', since there's only one transcript per participant."""
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured."
    try:
        q = client.table("participant_turn_annotations").delete().eq(
            "subject_id", subject_id
        ).eq("source", source)
        if source == "log":
            q = q.eq("log_filename", log_filename)
        q.execute()
        return True, None
    except Exception as e:
        return False, str(e)


def save_turn_annotations(
    subject_id: str, source: str, annotations: list[dict], model: str, log_filename: str = ""
) -> tuple[bool, str | None]:
    """Add these turn annotations for (subject_id, source[, log_filename]) —
    callers only pass turns that weren't already annotated (see
    annotated_turn_indexes / lib.annotate's skip_turn_indexes), so this is an
    append, not a replace. Upserts on the composite PK as a safety net against
    a duplicate call landing twice. `log_filename` is ignored (stays '') for
    source='transcript'."""
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured."
    if not annotations:
        return True, None
    lf = log_filename if source == "log" else ""
    try:
        client.table("participant_turn_annotations").upsert(
            [
                {"subject_id": subject_id, "source": source, "log_filename": lf,
                 "model": model, **a}
                for a in annotations
            ],
            on_conflict="subject_id,source,log_filename,turn_index",
        ).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def annotated_turn_indexes(subject_id: str, source: str, log_filename: str = "") -> set[int]:
    """Which turn_index values already have an annotation for (subject_id,
    source[, log_filename]) — used to skip already-annotated turns on a re-run."""
    client = get_supabase_client()
    if client is None:
        return set()
    try:
        q = (
            client.table("participant_turn_annotations")
            .select("turn_index").eq("subject_id", subject_id).eq("source", source)
        )
        if source == "log":
            q = q.eq("log_filename", log_filename)
        rows = q.execute().data or []
    except Exception:
        return set()
    return {r["turn_index"] for r in rows}


def list_turn_annotations(subject_id: str) -> list[dict]:
    """All turn annotations for one participant, ordered for display."""
    client = get_supabase_client()
    if client is None:
        return []
    try:
        rows = (
            client.table("participant_turn_annotations")
            .select("*").eq("subject_id", subject_id).execute().data or []
        )
    except Exception:
        return []
    return sorted(rows, key=lambda r: (r.get("source", ""), r.get("turn_index", 0)))


def list_all_turn_annotations() -> list[dict]:
    """[{subject_id, condition, source, phase, delegation_posture, trust_behavior}]
    — every turn annotation joined with condition, for the cohort tag-distribution
    charts."""
    client = get_supabase_client()
    if client is None:
        return []
    try:
        conds = {
            r["subject_id"]: r.get("condition")
            for r in (client.table("participants")
                      .select("subject_id,condition").execute().data or [])
        }
        rows = (
            client.table("participant_turn_annotations")
            .select("subject_id,source,phase,delegation_posture,trust_behavior")
            .execute().data or []
        )
    except Exception:
        return []
    return [{**r, "condition": conds.get(r["subject_id"])} for r in rows]


def list_notebook_section_scores() -> list[dict]:
    """[{subject_id, condition, section, score, max_pts, pct}] — one row per
    (participant, rubric section), for the notebook-assessment visualisations."""
    client = get_supabase_client()
    if client is None:
        return []
    try:
        conds = {
            r["subject_id"]: r.get("condition")
            for r in (client.table("participants")
                      .select("subject_id,condition").execute().data or [])
        }
        nbs = (
            client.table("participant_notebooks")
            .select("subject_id,results").execute().data or []
        )
    except Exception:
        return []
    out = []
    for nb in nbs:
        by_sec: dict[str, list[float]] = {}
        for it in nb.get("results") or []:
            s = (it.get("section") or "General").strip() or "General"
            sc, mx = float(it.get("score") or 0), float(it.get("max_pts") or 0)
            got = by_sec.setdefault(s, [0.0, 0.0])
            got[0] += sc
            got[1] += mx
        for s, (sc, mx) in by_sec.items():
            out.append({
                "subject_id": nb["subject_id"],
                "condition": conds.get(nb["subject_id"]),
                "section": s,
                "score": round(sc, 2),
                "max_pts": round(mx, 2),
                "pct": round(100 * sc / mx, 1) if mx else None,
            })
    return out


def list_graded_participant_notebooks() -> dict:
    """{subject_id: {"results", "total_score", "max_score"}} for every participant
    whose notebook has been graded. Shape matches build_report.records_from_graded."""
    client = get_supabase_client()
    if client is None:
        return {}
    try:
        rows = (
            client.table("participant_notebooks")
            .select("subject_id,results,total_score,max_score")
            .execute().data or []
        )
    except Exception:
        return {}
    return {
        r["subject_id"]: {
            "results": r["results"],
            "total_score": r.get("total_score"),
            "max_score": r.get("max_score"),
        }
        for r in rows if r.get("results")
    }


def list_pending_grading(include_graded: bool = False) -> list[dict]:
    """Participants whose notebook needs (re-)grading.

    By default only status pending/error. With include_graded=True, every
    participant with a notebook on file — so a newly uploaded/replaced rubric
    can be re-applied to notebooks that were already graded under the old one.
    Returns [{"subject_id", "notebook_text", "filename"}]."""
    client = get_supabase_client()
    if client is None:
        return []
    try:
        if include_graded:
            ids = None
        else:
            parts = (
                client.table("participants").select("subject_id,grading_status")
                .in_("grading_status", ["pending", "error"]).execute().data or []
            )
            ids = [p["subject_id"] for p in parts]
            if not ids:
                return []
        q = client.table("participant_notebooks").select("subject_id,filename,notebook_text")
        if ids is not None:
            q = q.in_("subject_id", ids)
        return q.execute().data or []
    except Exception:
        return []


def list_participant_survey_rows() -> list[dict]:
    """[{subject_id, condition, survey_type, elapsed_seconds, item_id, category,
    question, answer, time_spent_seconds}] — one row per (participant, survey
    item), for the cohort survey visualisations and the survey CSV export.

    Flattened here rather than in the chart layer so the admin gets the same
    long-format table it downloads."""
    client = get_supabase_client()
    if client is None:
        return []
    try:
        conds = {
            r["subject_id"]: r.get("condition")
            for r in (client.table("participants")
                      .select("subject_id,condition").execute().data or [])
        }
        surveys = (
            client.table("participant_surveys")
            .select("subject_id,survey_type,responses,elapsed_seconds").execute().data or []
        )
    except Exception:
        return []
    out = []
    for s in surveys:
        for r in s.get("responses") or []:
            out.append({
                "subject_id": s["subject_id"],
                "condition": conds.get(s["subject_id"]),
                "survey_type": s.get("survey_type"),
                "elapsed_seconds": s.get("elapsed_seconds"),
                "item_id": r.get("item_id"),
                "category": r.get("category"),
                "question": r.get("question"),
                "answer": r.get("answer"),
                "time_spent_seconds": r.get("time_spent_seconds"),
            })
    return out


# ---------------------------------------------------------------------------
# Task instructions (admin-authored, shown to participants)
# ---------------------------------------------------------------------------

def save_task_instruction(
    task_key: str, content: str, title: str | None = None
) -> tuple[bool, str | None]:
    """Upsert the instruction document for one task screen. `task_key` is one of
    lib.tasks.INSTRUCTION_KEYS; `content` is markdown shown to the participant
    verbatim."""
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured (SUPABASE_URL/SUPABASE_KEY not set)."
    try:
        client.table("task_instructions").upsert(
            {
                "task_key": task_key,
                "title": (title or "").strip() or None,
                "content": content,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="task_key",
        ).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def get_task_instruction(task_key: str) -> dict | None:
    """One task's instruction document, or None if it hasn't been uploaded."""
    client = get_supabase_client()
    if client is None or not task_key:
        return None
    try:
        rows = (
            client.table("task_instructions").select("*")
            .eq("task_key", task_key).limit(1).execute().data or []
        )
    except Exception:
        return None
    return rows[0] if rows else None


def list_task_instructions() -> dict[str, dict]:
    """{task_key: row} for every uploaded instruction document."""
    client = get_supabase_client()
    if client is None:
        return {}
    try:
        rows = client.table("task_instructions").select("*").execute().data or []
    except Exception:
        return {}
    return {r["task_key"]: r for r in rows}


def delete_task_instruction(task_key: str) -> tuple[bool, str | None]:
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured."
    try:
        client.table("task_instructions").delete().eq("task_key", task_key).execute()
        return True, None
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Per-task timing
# ---------------------------------------------------------------------------

def start_participant_task(
    subject_id: str, task_key: str, limit_seconds: int | None
) -> dict | None:
    """Mark a timed task as started and return its timing row.

    Deliberately does NOT move an existing started_at: a participant who
    refreshes the page, or comes back to a task screen, resumes the clock they
    already started rather than getting a fresh allowance."""
    client = get_supabase_client()
    if client is None:
        return None
    existing = get_participant_task_timing(subject_id, task_key)
    if existing is not None:
        return existing
    try:
        client.table("participant_task_timings").upsert(
            {
                "subject_id": subject_id,
                "task_key": task_key,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "limit_seconds": limit_seconds,
            },
            on_conflict="subject_id,task_key",
        ).execute()
    except Exception:
        return None
    return get_participant_task_timing(subject_id, task_key)


def finish_participant_task(subject_id: str, task_key: str) -> tuple[bool, str | None]:
    """Stamp finished_at. Re-finishing (e.g. a re-upload) moves it, so the
    recorded span always covers everything the participant did on that task."""
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured."
    try:
        client.table("participant_task_timings").update(
            {"finished_at": datetime.now(timezone.utc).isoformat()}
        ).eq("subject_id", subject_id).eq("task_key", task_key).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def get_participant_task_timing(subject_id: str, task_key: str) -> dict | None:
    client = get_supabase_client()
    if client is None:
        return None
    try:
        rows = (
            client.table("participant_task_timings").select("*")
            .eq("subject_id", subject_id).eq("task_key", task_key)
            .limit(1).execute().data or []
        )
    except Exception:
        return None
    return rows[0] if rows else None


def list_participant_task_timings(subject_id: str) -> dict[str, dict]:
    """{task_key: timing row} for one participant."""
    client = get_supabase_client()
    if client is None:
        return {}
    try:
        rows = (
            client.table("participant_task_timings").select("*")
            .eq("subject_id", subject_id).execute().data or []
        )
    except Exception:
        return {}
    return {r["task_key"]: r for r in rows}


def list_all_task_timings() -> list[dict]:
    """Every participant's task timings, for the cohort view."""
    client = get_supabase_client()
    if client is None:
        return []
    try:
        return client.table("participant_task_timings").select("*").execute().data or []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Extra main-task files
# ---------------------------------------------------------------------------

def save_participant_extra_file(
    subject_id: str, filename: str, raw: bytes
) -> tuple[bool, str | None]:
    """Store one optional main-task attachment under its own name. Text is kept
    verbatim; anything that isn't valid UTF-8 is base64'd so binaries (figures,
    exports) survive a round trip."""
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured."
    try:
        content, encoding = raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        content, encoding = base64.b64encode(raw).decode("ascii"), "base64"
    try:
        client.table("participant_extra_files").upsert(
            {
                "subject_id": subject_id,
                "filename": filename,
                "content": content,
                "encoding": encoding,
                "byte_size": len(raw),
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="subject_id,filename",
        ).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def list_participant_extra_files(subject_id: str) -> list[dict]:
    client = get_supabase_client()
    if client is None:
        return []
    try:
        return (
            client.table("participant_extra_files").select("*")
            .eq("subject_id", subject_id).execute().data or []
        )
    except Exception:
        return []


def delete_participant_extra_file(subject_id: str, filename: str) -> tuple[bool, str | None]:
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured."
    try:
        client.table("participant_extra_files").delete().eq(
            "subject_id", subject_id
        ).eq("filename", filename).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def extra_file_bytes(row: dict) -> bytes:
    """The original bytes of a stored attachment, whichever way it was encoded."""
    content = row.get("content") or ""
    if (row.get("encoding") or "utf-8") == "base64":
        return base64.b64decode(content)
    return content.encode("utf-8")


# ---------------------------------------------------------------------------
# Resume point
# ---------------------------------------------------------------------------

def set_participant_stage(
    subject_id: str, stage: str, status: str | None = None
) -> tuple[bool, str | None]:
    """Record which screen the participant is on, so re-entering their subject ID
    resumes exactly there. `status` is kept in step for the Admin overview."""
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured."
    payload: dict = {"stage": stage}
    if status is not None:
        payload["status"] = status
    try:
        client.table("participants").update(payload).eq("subject_id", subject_id).execute()
        return True, None
    except Exception as e:
        return False, str(e)
