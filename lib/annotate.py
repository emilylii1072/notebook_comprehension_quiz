"""Auto-annotate participant turns: one LLM call per turn, for the
delegation-conditions study. Two sources, two different taxonomies:

  - a session log's User Prompt events (`lib.timeline.parse_jsonl` output) — one
    call per prompt, tagging the *behaviour* (phase / delegation posture / trust),
    with same-turn Agent Call / Tool Call text as context when the log format has
    it ("transcript" format; "history" format has no context at all).
  - a verbal-assessment transcript's Q/A pairs (`lib.transcript.parse_transcript`
    output) — the participant answering interview questions about the notebook
    task *after the fact*. There's no delegation happening here; what matters is
    whether the spoken answer is *truthful to the participant's own notebook* —
    one call per pair, checked against the notebook text (ground truth).

Tags are constrained enums (Pydantic `Literal`s), so `client.messages.parse`
validates structure for us — no self-check/repair loop needed, unlike quiz.py's
free-text grading problem.
"""

from typing import Literal

import anthropic
from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

MODEL = "claude-opus-5"

_SYSTEM = (
    "You annotate one turn from a study of AI-delegation behaviour. Participants "
    "used Claude Code to do a data-science task under one of several delegation "
    "conditions. For the given turn (and any context showing what happened right "
    "after it), tag:\n\n"
    "- phase: what kind of work this turn is doing — planning (deciding what to do "
    "next / scoping), implementing (asking for code/changes to be written), "
    "debugging (fixing an error or unexpected result), verifying (checking/testing "
    "output), or reflecting (explaining reasoning, summarising, or discussing "
    "results without asking for new work).\n"
    "- delegation_posture: how much the participant is handing off vs. directing — "
    "high_level_ask (\"just fix it\", \"build the model\", open-ended), step_by_step "
    "(specific, detailed instructions), or clarifying_question (asking the "
    "assistant something rather than instructing it).\n"
    "- trust_behavior: how the participant responds to what came right after this "
    "turn — accepts_unreviewed (moves on without comment), reviews_edits (checks, "
    "questions, or modifies the result), rejects_redirects (rejects it or changes "
    "direction). If no context about what happened next is given, or the context "
    "gives no real signal either way, you MUST answer insufficient_context — never "
    "guess.\n\n"
    "Write one short sentence of reasoning citing what in the turn/context drove "
    "each tag."
)

_INSTRUCTIONS = """\
TURN:
%s

CONTEXT (what happened right after this turn, if known):
%s
"""


class TurnAnnotation(BaseModel):
    phase: Literal["planning", "implementing", "debugging", "verifying", "reflecting"]
    delegation_posture: Literal["high_level_ask", "step_by_step", "clarifying_question"]
    trust_behavior: Literal[
        "accepts_unreviewed", "reviews_edits", "rejects_redirects", "insufficient_context"
    ]
    reasoning: str


def annotate_turn(client: Anthropic, turn_text: str, context: str | None) -> TurnAnnotation:
    """One LLM call, tagging a single turn. Raises RuntimeError with a user-facing
    message on a malformed / refused / truncated response."""
    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=1024,
            timeout=600.0,
            system=_SYSTEM,
            messages=[{
                "role": "user",
                "content": _INSTRUCTIONS % (turn_text or "(empty)", context or "(none)"),
            }],
            output_format=TurnAnnotation,
        )
    except ValidationError as e:
        raise RuntimeError(f"The model returned a malformed response: {e} — try again.") from e
    if response.stop_reason == "refusal":
        detail = getattr(response.stop_details, "explanation", None) or "safety refusal"
        raise RuntimeError(f"The model declined to annotate this turn ({detail}).")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("The model ran out of room before finishing — try again.")
    parsed = response.parsed_output
    if parsed is None:
        raise RuntimeError("The model returned an unparseable response — try again.")
    return parsed


def _log_context(events: list[dict], turn: int, this_idx: int) -> str | None:
    """Agent Call / Tool Call text from the same turn, in order, after this event."""
    pieces = []
    for e in events:
        if e["turn"] != turn or e["idx"] <= this_idx:
            continue
        if e["lane"] == "Agent Call" and e.get("text"):
            pieces.append(e["text"])
        elif e["lane"] == "Tool Call":
            pieces.append(f"[used tool {e.get('tool_name')}] {e.get('result_text', '')[:500]}")
    return "\n".join(pieces) or None


def annotate_log(
    client: Anthropic, parsed: dict, skip_turn_indexes: frozenset[int] = frozenset()
) -> list[dict]:
    """Tag every not-yet-annotated User Prompt event in a parsed session log (either
    format) — turns whose `turn` number is in `skip_turn_indexes` are left alone, so
    re-running only fills in what's missing. Returns [{"turn_index", "turn_text",
    "phase", "delegation_posture", "trust_behavior", "reasoning"}, ...]. Raises on
    the first failed call — callers should catch and report per-participant, same
    as parse_transcript."""
    events = parsed.get("events", [])
    is_transcript_format = parsed.get("format") == "transcript"
    out = []
    for e in events:
        if e["lane"] != "User Prompt" or e["turn"] in skip_turn_indexes:
            continue
        context = _log_context(events, e["turn"], e["idx"]) if is_transcript_format else None
        tag = annotate_turn(client, e["text"], context)
        out.append({"turn_index": e["turn"], "turn_text": e["text"], **tag.model_dump()})
    return out


_TRANSCRIPT_SYSTEM = (
    "You fact-check one answer from a verbal assessment given right after a "
    "take-home data-science task. The interviewer asked the participant questions "
    "about the task; you're given the participant's own notebook as ground truth. "
    "Judge whether the spoken answer is truthful to what the notebook actually "
    "shows — not whether the answer is a *good* approach, just whether it "
    "accurately describes what the participant actually did.\n\n"
    "Tag accuracy:\n"
    "- accurate: the answer's factual claims match what the notebook shows.\n"
    "- partially_accurate: some claims match; others are vague, overstated, or "
    "slightly off from what the notebook actually contains.\n"
    "- inaccurate: the answer contradicts or fabricates something the notebook "
    "does not show (e.g. describing a method, metric, or step that isn't there).\n"
    "- unverifiable: the question is conceptual/opinion-based (e.g. \"what would "
    "you do differently\") and isn't a claim about the notebook's contents at all.\n\n"
    "Write one short sentence of reasoning citing the specific notebook content "
    "(or its absence) that the verdict rests on."
)

_TRANSCRIPT_INSTRUCTIONS = """\
QUESTION: %s

PARTICIPANT'S ANSWER: %s

THEIR NOTEBOOK (ground truth):
%s
"""


class TranscriptAnswerAnnotation(BaseModel):
    accuracy: Literal["accurate", "partially_accurate", "inaccurate", "unverifiable"]
    reasoning: str


def annotate_transcript_answer(
    client: Anthropic, question: str, answer: str, notebook_text: str
) -> TranscriptAnswerAnnotation:
    """One LLM call, checking one verbal-assessment answer against the
    participant's own notebook. Same error contract as annotate_turn."""
    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=1024,
            timeout=600.0,
            system=_TRANSCRIPT_SYSTEM,
            messages=[{
                "role": "user",
                "content": _TRANSCRIPT_INSTRUCTIONS % (
                    question or "(empty)", answer or "(no answer given)",
                    notebook_text or "(no notebook on file)",
                ),
            }],
            output_format=TranscriptAnswerAnnotation,
        )
    except ValidationError as e:
        raise RuntimeError(f"The model returned a malformed response: {e} — try again.") from e
    if response.stop_reason == "refusal":
        detail = getattr(response.stop_details, "explanation", None) or "safety refusal"
        raise RuntimeError(f"The model declined to check this answer ({detail}).")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("The model ran out of room before finishing — try again.")
    parsed = response.parsed_output
    if parsed is None:
        raise RuntimeError("The model returned an unparseable response — try again.")
    return parsed


def annotate_transcript(
    client: Anthropic,
    pairs: list[dict],
    notebook_text: str,
    skip_turn_indexes: frozenset[int] = frozenset(),
) -> list[dict]:
    """Fact-check every not-yet-annotated {timestamp, question, answer} pair from a
    verbal-assessment transcript against the participant's own notebook (pair
    list-index in `skip_turn_indexes` is left alone)."""
    out = []
    for i, pair in enumerate(pairs):
        if i in skip_turn_indexes:
            continue
        tag = annotate_transcript_answer(
            client, pair.get("question", ""), pair.get("answer", ""), notebook_text
        )
        out.append({"turn_index": i, "turn_text": pair.get("question", ""), **tag.model_dump()})
    return out


def call_with_errors_surfaced(fn, *args, **kwargs):
    """Run an API-calling function, converting SDK errors to readable messages.
    (Same contract as lib.grading.call_with_errors_surfaced.)"""
    import streamlit as st

    try:
        return fn(*args, **kwargs)
    except anthropic.AuthenticationError:
        st.error("Authentication failed. Set ANTHROPIC_API_KEY and reboot the app.")
    except anthropic.RateLimitError:
        st.error("Rate limited (or out of quota) — check your Anthropic plan/billing and try again.")
    except anthropic.APIStatusError as e:
        st.error(f"API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        st.error("Could not reach the Anthropic API — check your network connection.")
    except RuntimeError as e:
        st.error(str(e))
    return None
