"""Autograde the debugging task with Claude Opus 5.

Participants are handed a section of a coworker's notebook that "performs too well"
and write a ranked list of the bugs they find, each with a location and a fix. The
admin uploads that buggy notebook (Admin > Task instructions); this module turns it
into an answer key and grades each participant's write-up against the key.

The answer key is one shared, admin-editable list, not something re-derived per
participant: if every write-up were graded against whatever bugs the model happened
to spot that time, the number of bugs — and so the maximum score — could differ
between participants and the scores wouldn't be comparable.
"""

from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

from lib.annotate import call_with_errors_surfaced  # noqa: F401  (re-exported for callers)

MODEL = "claude-opus-5"
MAX_PTS_PER_BUG = 2


class KeyBug(BaseModel):
    title: str
    location: str
    problem: str
    fix: str


class AnswerKey(BaseModel):
    bugs: list[KeyBug]


class BugGrade(BaseModel):
    bug: str
    score: int
    reasoning: str


class DebugGrade(BaseModel):
    items: list[BugGrade]
    notes: str


def _parse(client: Anthropic, system, user: str, output_format, max_tokens: int, what: str):
    """One structured call with the same error contract as lib.transcript."""
    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=max_tokens,
            timeout=600.0,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=output_format,
        )
    except ValidationError as e:
        raise RuntimeError(f"The model returned a malformed response: {e} — try again.") from e
    if response.stop_reason == "refusal":
        detail = getattr(response.stop_details, "explanation", None) or "safety refusal"
        raise RuntimeError(f"The model declined to {what} ({detail}).")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("The model ran out of room before finishing — try again.")
    if response.parsed_output is None:
        raise RuntimeError("The model returned an unparseable response — try again.")
    return response.parsed_output


_TASK_CONTEXT = (
    "Study context: participants did a data-science task (an MVP employee-attrition "
    "model). In the debugging task they were given a section of a coworker's Python "
    "notebook for that same task, with most unrelated code omitted, and were told the "
    "coworker feels something is off because the model's performance is too good. "
    "They had 15 minutes to manually inspect it for bugs (flaws, areas for "
    "improvement, unclarities) and possible solutions, and to rank them in a "
    "markdown file."
)


_KEY_SYSTEM = (
    "You are an expert data scientist reviewing a notebook that a research study "
    "gave participants to debug.\n\n" + _TASK_CONTEXT + "\n\n"
    "List the genuine bugs in the notebook below — the flaws that make its results "
    "misleading or invalid (for example data leakage, how the target/label is "
    "constructed, how the data is split, how it is evaluated, class imbalance) — "
    "most consequential first. For each: a short title, the cell number(s) where it "
    "occurs, what is wrong and why it matters, and the fix. Only list real, "
    "defensible bugs that you can point to in the code; do not pad the list with "
    "style nits or speculation, and do not invent bugs. An admin will review and "
    "edit your list before it is used to grade anyone."
)


def generate_answer_key(client: Anthropic, buggy_notebook_text: str) -> str:
    """Draft the answer key for the buggy notebook, as markdown the admin edits and
    saves. Cell numbers refer to the `--- ... cell N ---` markers of the flattened
    notebook (lib.notebook.notebook_to_text)."""
    key = _parse(
        client, _KEY_SYSTEM,
        f"===== BEGIN NOTEBOOK =====\n{buggy_notebook_text}\n===== END NOTEBOOK =====",
        AnswerKey, max_tokens=8000, what="review this notebook",
    )
    if not key.bugs:
        raise RuntimeError("The model found no bugs in this notebook — check that it is the buggy one.")
    return "\n\n".join(
        f"{i}. **{b.title.strip()}** — {b.location.strip()}\n"
        f"   - Problem: {b.problem.strip()}\n"
        f"   - Fix: {b.fix.strip()}"
        for i, b in enumerate(key.bugs, 1)
    )


def _grade_system(buggy_notebook_text: str, answer_key: str) -> str:
    return (
        "You grade one participant's debugging write-up against a fixed answer key, "
        "strictly and consistently.\n\n" + _TASK_CONTEXT + "\n\n"
        "## The buggy notebook they were given\n"
        f"{buggy_notebook_text}\n\n"
        "## Answer key (the bugs to look for)\n"
        f"{answer_key}\n\n"
        "## Grading rules\n"
        "Return one item per bug in the answer key, in the key's order, copying the "
        "bug's title into `bug`. Score each on a 2-point scale:\n"
        "- 2: the write-up identifies this bug (the right problem, in the right place) "
        "AND gives a fix that would actually resolve it.\n"
        "- 1: partly there — it names the problem only vaguely or in the wrong place, "
        "or it diagnoses it correctly but gives no fix or one that would not resolve it.\n"
        "- 0: the bug is not identified (or an empty write-up).\n"
        "Credit substance, not wording; a cell number that is a little off is fine if "
        "the code being pointed at is clearly the right code. The order the participant "
        "ranked their findings in is not graded. Reports that are not in the answer key "
        "earn nothing and cost nothing — mention in `notes`, briefly, any reported "
        "issue that is wrong or is a valid extra bug, or leave `notes` empty. Each "
        "`reasoning` is one concrete sentence."
    )


def grade_debug_writeup(
    client: Anthropic, buggy_notebook_text: str, answer_key: str, writeup_md: str
) -> dict:
    """Grade one participant's debug write-up against the answer key. Returns
    {"results": [{"section","criterion","max_pts","score","reasoning"}, ...],
    "notes": str} — one result per key bug, scored 0-2 (clamped)."""
    graded = _parse(
        client,
        # Notebook + key are identical for every participant, so cache them.
        [{
            "type": "text",
            "text": _grade_system(buggy_notebook_text, answer_key),
            "cache_control": {"type": "ephemeral"},
        }],
        f"===== BEGIN WRITE-UP =====\n{writeup_md.strip() or '(empty)'}\n===== END WRITE-UP =====",
        DebugGrade, max_tokens=8000, what="grade this write-up",
    )
    if not graded.items:
        raise RuntimeError("The model returned no graded items — try again.")
    results = [
        {
            "section": "Debugging",
            "criterion": g.bug.strip(),
            "max_pts": MAX_PTS_PER_BUG,
            "score": max(0, min(int(g.score), MAX_PTS_PER_BUG)),
            "reasoning": g.reasoning,
        }
        for g in graded.items
    ]
    return {"results": results, "notes": graded.notes.strip()}
