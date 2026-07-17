"""Shared Supabase (Postgres) integration for the Dynamic Evaluation Tools app.

Credentials are read from `st.secrets` first (Streamlit Community Cloud's secrets
manager) and fall back to environment variables (populated from a local `.env` via
python-dotenv, for local dev). If neither is configured, `get_supabase_client()`
returns None and callers should degrade gracefully -- the app must keep working
without a database attached.
"""

import os
import random

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
