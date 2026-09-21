"""Autograde the ideation task with Claude Opus 5.

Participants upload nothing for this task: they pitch their best idea aloud to a
"Product Manager", the admin uploads a transcript of that pitch (Admin > Participant
> Ideate), and it is scored here against an admin-uploaded rubric (Admin > Task
instructions). As with the notebook rubric, the rubric text is handed to the model
VERBATIM, so whatever columns / guidance / point values it carries are used as
written; this module invents no criteria of its own.
"""

from anthropic import Anthropic

from lib.grading import MODEL, grade_with_prompt  # noqa: F401  (MODEL re-exported for callers)

_INSTRUCTIONS = """\
- Score every scoreable rubric item from 0 to that item's maximum points, reading
  every part of the rubric — point values and any full/partial/low-credit guidance —
  as written.
- Judge only what the participant said in the transcript. Do not give credit for
  things they might have meant but did not say.
- The transcript is speech-to-text of a spoken pitch: ignore filler words, false
  starts, and obvious transcription errors, and judge the substance of the idea, not
  the delivery — unless the rubric itself says to score delivery.
- Use each item's scoring guidance on a consistent scale: 0 points when the
  criterion is missing or completely off; roughly half of the maximum when the
  partial-credit guidance describes the pitch; the maximum only when the full-credit
  guidance is clearly met. Interpolate between anchors for in-between cases.
- Where an item says some evidence is scored under a different item, do NOT
  double-credit it here.
- An empty or near-empty transcript should receive 0s with a brief explanation.
- Be consistent: the same evidence must always earn the same score.
- Reasoning: 1-3 concrete sentences, quoting or closely paraphrasing the pitch."""


def build_system_prompt(task_text: str, rubric: str) -> str:
    return f"""You are an expert data-science and product reviewer grading spoken pitches \
from a research study, strictly and consistently against a fixed rubric.


## The task the participant was given
{task_text.strip() or "(not available)"}

They pitched the idea they picked to a Product Manager, out loud. You are grading the \
transcript of that pitch.


## Rubric (exactly as the researcher uploaded it)
```
{rubric}
```


## Grading rules
{_INSTRUCTIONS}

## Output format
One entry per scoreable rubric item, in the order they appear in the rubric. Copy each \
item's section name (if it has one) and maximum points from the rubric verbatim."""


def grade_pitch(
    client: Anthropic, task_text: str, rubric: str, transcript_text: str
) -> list[dict]:
    """Score one pitch transcript against the rubric. Returns
    [{"section","criterion","max_pts","score","reasoning"}]; raises RuntimeError with
    a user-facing message on a malformed / refused / truncated response."""
    user_content = (
        "Grade the following pitch transcript.\n\n"
        "===== BEGIN PITCH TRANSCRIPT =====\n"
        f"{transcript_text.strip() or '(empty)'}\n"
        "===== END PITCH TRANSCRIPT ====="
    )
    return grade_with_prompt(
        client, build_system_prompt(task_text, rubric), user_content, "pitch"
    )
