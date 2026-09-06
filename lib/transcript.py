"""Parse a verbal-assessment transcript into timestamped question/answer pairs.

The admin uploads a plain-text transcript per participant. It looks like:

    00:00
    Okay. Now you have some response questions based on the task you just performed?

    00:41
    OK, first question. If we were to create an employee attrition model, what would
    it look like? You get the input data like the pq_data set and you feed ...

Each `MM:SS` line marks a segment; inside a segment an interviewer's questions and
the participant's answers run together with no speaker labels, and one segment may
hold several question/answer pairs. `parse_transcript` uses Claude Opus 5 to split
it. It does NOT score anything.
"""

import anthropic
from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

MODEL = "claude-opus-5"

_SYSTEM = (
    "You clean up transcripts of a short verbal assessment given right after a "
    "take-home data-science task: building an MVP employee-attrition model on the "
    "Viva Insights `pq_data` sample dataset. Knowing that context, you can correct "
    "obvious speech-to-text word errors (e.g. 'nutrition' -> 'attrition', 'FRQ', "
    "'pq data' -> 'pq_data')."
)

_INSTRUCTIONS = """\
Split the transcript below into question/answer pairs.

- `question`: the interviewer's question. The interviewer's questions are standard and
  clear -- lightly clean speech-to-text errors and phrasing, but keep the meaning.
- `answer`: the participant's spoken response to that question, VERBATIM. Fix only
  unambiguous word-level transcription errors; never rephrase, summarise, condense,
  or improve the answer. If the participant gave no real answer, use "".
- `timestamp`: the `MM:SS` label of the segment the question appears in (a segment
  can contain several pairs -- they share that segment's timestamp).
- Keep the pairs in spoken order. Skip pure preamble or sign-off lines that are not a
  question ("Now you have some response questions...", "that's it for the FRQs").
- Do not invent questions or answers that are not in the transcript.

TRANSCRIPT:
%s
"""


class TranscriptQA(BaseModel):
    timestamp: str
    question: str
    answer: str


class ParsedTranscript(BaseModel):
    pairs: list[TranscriptQA]


def parse_transcript(client: Anthropic, raw_text: str) -> list[dict]:
    """Return [{"timestamp", "question", "answer"}, ...]. Raises RuntimeError with a
    user-facing message on a malformed / refused / truncated response."""
    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=16000,
            timeout=600.0,
            system=_SYSTEM,
            messages=[{"role": "user", "content": _INSTRUCTIONS % raw_text}],
            output_format=ParsedTranscript,
        )
    except ValidationError as e:
        raise RuntimeError(f"The model returned a malformed response: {e} — try again.") from e
    if response.stop_reason == "refusal":
        detail = getattr(response.stop_details, "explanation", None) or "safety refusal"
        raise RuntimeError(f"The model declined to parse this transcript ({detail}).")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("The model ran out of room before finishing — try again.")
    parsed = response.parsed_output
    if parsed is None:
        raise RuntimeError("The model returned an unparseable response — try again.")
    return [p.model_dump() for p in parsed.pairs]


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
