"""Shared Supabase (Postgres) integration for the Dynamic Evaluation Tools app.

Credentials are read from `st.secrets` first (Streamlit Community Cloud's secrets
manager) and fall back to environment variables (populated from a local `.env` via
python-dotenv, for local dev). If neither is configured, `get_supabase_client()`
returns None and callers should degrade gracefully -- the app must keep working
without a database attached.
"""

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

def save_rubric(name: str, task: str, rubric_csv: str) -> tuple[bool, str | None]:
    """Upsert a rubric (task + the raw uploaded CSV) by name. Returns (success, error)."""
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured (SUPABASE_URL/SUPABASE_KEY not set)."
    try:
        client.table("grading_rubric").upsert(
            {
                "name": name,
                "task": task,
                "rubric_csv": rubric_csv,
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
            .select("name,task,rubric_csv")
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
                      "participant_quiz", "participant_logs"):
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


def save_participant_log(
    subject_id: str, filename: str, raw_jsonl: str, metrics: dict
) -> tuple[bool, str | None]:
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured."
    try:
        client.table("participant_logs").upsert(
            {
                "subject_id": subject_id,
                "filename": filename,
                "raw_jsonl": raw_jsonl,
                "metrics": metrics,
            },
            on_conflict="subject_id",
        ).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def mark_participant_stage1_done(subject_id: str) -> tuple[bool, str | None]:
    """Part 1 (notebook + quiz + grading) is done; Part 2 (docs + log) still owed.
    No-op on a participant already marked complete."""
    client = get_supabase_client()
    if client is None:
        return False, "Database not configured."
    try:
        (
            client.table("participants")
            .update({"status": "quiz_done"})
            .eq("subject_id", subject_id)
            .neq("status", "complete")
            .execute()
        )
        return True, None
    except Exception as e:
        return False, str(e)


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
        "log": None, "transcript": None,
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
        lg = (
            client.table("participant_logs").select("*")
            .eq("subject_id", subject_id).limit(1).execute().data or []
        )
        out["log"] = lg[0] if lg else None
        tr = (
            client.table("participant_transcripts").select("*")
            .eq("subject_id", subject_id).limit(1).execute().data or []
        )
        out["transcript"] = tr[0] if tr else None
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
        logs = {
            r["subject_id"]: r
            for r in (client.table("participant_logs")
                      .select("subject_id,metrics").execute().data or [])
        }
        transcript_ids = {
            r["subject_id"]
            for r in (client.table("participant_transcripts")
                      .select("subject_id").execute().data or [])
        }
    except Exception:
        return []

    rows = []
    for p in parts:
        sid = p["subject_id"]
        nb = notebooks.get(sid) or {}
        qz = quizzes.get(sid) or {}
        metrics = (logs.get(sid) or {}).get("metrics") or {}
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
                "session_duration_s": metrics.get("session_duration_s"),
                "n_tool_calls": metrics.get("n_tool_calls"),
                "n_edits": metrics.get("n_edits"),
                "time_to_first_tool_call_s": metrics.get("time_to_first_tool_call_s"),
                "median_inter_tool_gap_s": metrics.get("median_inter_tool_gap_s"),
                "chat_count": metrics.get("chat_count"),
                "instruct_count": metrics.get("instruct_count"),
                "has_transcript": sid in transcript_ids,
            }
        )
    return rows


def list_pending_grading() -> list[dict]:
    """Participants whose notebook still needs grading (status pending/error).
    Returns [{"subject_id", "notebook_text", "filename"}]."""
    client = get_supabase_client()
    if client is None:
        return []
    try:
        parts = (
            client.table("participants").select("subject_id,grading_status")
            .in_("grading_status", ["pending", "error"]).execute().data or []
        )
        ids = [p["subject_id"] for p in parts]
        if not ids:
            return []
        nbs = (
            client.table("participant_notebooks")
            .select("subject_id,filename,notebook_text")
            .in_("subject_id", ids).execute().data or []
        )
        return nbs
    except Exception:
        return []
