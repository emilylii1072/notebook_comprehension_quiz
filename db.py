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


def _get_secret(key: str) -> str | None:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass  # no secrets.toml at all -- st.secrets raises rather than returning {}
    return os.environ.get(key)


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
