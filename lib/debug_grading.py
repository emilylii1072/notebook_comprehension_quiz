"""Autograde the debugging task with Claude Opus 5.

Participants are handed a section of a coworker's notebook that "performs too well"
and write a ranked list of the bugs they find, each with a location and a fix. The
admin uploads that buggy notebook (Admin > Task instructions); Claude reads it and
the participant's write-up, pulls out every bug the participant reported, and scores
each one:

  +1  the reported bug is a genuine bug in the notebook
  +1  the proposed fix is a valid fix for it

There is no answer key: the model judges each reported bug on its own merits, so the
maximum score is 2 x the number of bugs the participant reported.
"""

from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

from lib.annotate import call_with_errors_surfaced  # noqa: F401  (re-exported for callers)

MODEL = "claude-opus-5"
MAX_PTS_PER_BUG = 2


class BugGrade(BaseModel):
    bug: str              # short title of the bug as the participant reported it
    cell: int | None       # which notebook cell it refers to (0-based, "Cell N"), or null
    is_real_bug: bool     # +1
    fix_is_valid: bool    # +1
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


def _grade_system(buggy_notebook_text: str) -> str:
    return (
        "You are an expert data scientist grading one participant's debugging "
        "write-up, strictly and consistently.\n\n" + _TASK_CONTEXT + "\n\n"
        "## The notebook they were given\n"
        "Cell numbers refer to the `--- ... cell N ---` markers.\n"
        f"{buggy_notebook_text}\n\n"
        "## Grading rules\n"
        "Read the write-up and list every distinct bug the participant reported, in "
        "the order they wrote them, one item per bug (if two entries describe the "
        "same underlying bug, count it once; if one entry bundles several separate "
        "bugs, split it). Copy or condense the participant's own title for the bug "
        "into `bug`. Set `cell` to the N of the single cell (`--- ... cell N ---`) "
        "the bug most centrally concerns — the write-up's own stated location if it "
        "gives one and it checks out, otherwise the cell you'd point to yourself; "
        "null only if no single cell is clearly the right one (the bug spans "
        "several cells, or nothing in the notebook matches). Then score each item "
        "on two independent yes/no points:\n"
        "- `is_real_bug` (+1): the reported issue is a genuine bug in the notebook — "
        "a flaw you can point to in the code that makes its results misleading or "
        "invalid (data leakage, how the target/label is built, how the data is split, "
        "how it is evaluated, class imbalance, and so on). No: it is not actually "
        "wrong, is only a style nit or speculation, or does not apply to this code.\n"
        "- `fix_is_valid` (+1): the fix the participant proposed would actually "
        "resolve that bug. No: there is no fix, it is too vague to act on, or it would "
        "not resolve the problem. Always no if `is_real_bug` is false.\n"
        "Credit substance, not wording; a cell number that is a little off is fine if "
        "the code being pointed at is clearly the right code. Do not reward the "
        "ranking. Judge each report on its own merits — do not penalise a valid bug "
        "for being one you would not have listed. Each `reasoning` is one concrete "
        "sentence covering both the bug and the fix. Use `notes` only for a brief "
        "remark that does not fit the items, or leave it empty. If the write-up "
        "reports no bugs, return no items."
    )


def grade_debug_writeup(client: Anthropic, buggy_notebook_text: str, writeup_md: str) -> dict:
    """Grade one participant's debug write-up. Returns
    {"results": [{"section","criterion","max_pts","score","reasoning","cell"}, ...],
    "notes": str} — one result per bug the participant reported, scored 0-2
    (+1 real bug, +1 valid fix). `cell` is the 0-based notebook cell (matching
    notebook_to_html's "Cell N" labels) the bug refers to, or None if the model
    couldn't tie it to one specific cell -- Admin's Debug sub-tab uses it to show
    each bug alongside the cell it's about."""
    if not writeup_md.strip():
        return {"results": [], "notes": "The write-up is empty."}
    graded = _parse(
        client,
        # The notebook is identical for every participant, so cache it.
        [{
            "type": "text",
            "text": _grade_system(buggy_notebook_text),
            "cache_control": {"type": "ephemeral"},
        }],
        f"===== BEGIN WRITE-UP =====\n{writeup_md.strip()}\n===== END WRITE-UP =====",
        DebugGrade, max_tokens=8000, what="grade this write-up",
    )
    results = []
    for g in graded.items:
        real = bool(g.is_real_bug)
        fix = real and bool(g.fix_is_valid)
        results.append({
            "section": "Debugging",
            "criterion": g.bug.strip(),
            "max_pts": MAX_PTS_PER_BUG,
            "score": int(real) + int(fix),
            "reasoning": f"Bug: {'✓' if real else '✗'} · Fix: {'✓' if fix else '✗'} — {g.reasoning.strip()}",
            "cell": g.cell,
        })
    return {"results": results, "notes": graded.notes.strip()}
