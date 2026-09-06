"""Grade one Jupyter notebook against a rubric CSV with Claude Opus 5.

The rubric CSV is handed to the model VERBATIM (never reparsed or transformed), so
every column of grading guidance is used as-is. `grade_notebook` returns one score +
reasoning per rubric item; the `build_*_df` / `build_excel` helpers turn a set of
graded notebooks into summary/detail tables.

Lifted from the old Notebook Grader page; the Streamlit tab shell now lives in
app_pages/admin.py.
"""

import io

import anthropic
import pandas as pd
import streamlit as st
from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

from lib.llm import get_client  # noqa: F401  (re-exported for callers that import it from here)

MODEL = "claude-opus-5"  # swap to "claude-sonnet-5" for cheaper/faster test iteration
MAX_TOKENS = 32000
DEFAULT_RUBRIC_NAME = "attrition_v1"

# The default task graded against — the employee-attrition take-home. Stored to
# Supabase alongside the rubric so every notebook is graded against the exact task
# text that was in effect.
DEFAULT_TASK = """\
Participants had ~30 minutes and were told this is an open-ended, intentionally
underspecified research task (identifying gaps and making reasonable assumptions
is part of the task). The brief:

  A PM asks for an MVP employee attrition model using Viva Insights data, to
  inform whether the feature is worth engineering investment. PM questions:
  (1) If we created an employee attrition model, what would it look like?
  (2) What would be the inputs and outputs of the model?
  (3) What factors matter when bringing this model into production?

  Data: the sample Person Query dataset (pq_data) from the `vivainsights`
  Python package. The brief explicitly notes the dataset is incomplete: an
  attrition label column is missing and must be engineered or simulated.

  Required output: a Jupyter notebook with (a) recommended model algorithm and
  rationale, (b) code demonstrating the model end-to-end, (c) example outputs
  (predictions, feature importance, evaluation metrics), (d) a brief discussion
  of production considerations. Participants worked with an AI coding agent.
"""


def build_system_prompt(task: str, rubric_csv: str) -> str:
    """The grader's system message: study context, the rubric CSV verbatim, and the
    grading rules. The CSV is embedded as uploaded — never reparsed or transformed."""
    return f"""You are an expert data-science instructor grading Jupyter notebooks \
from a research study, strictly and consistently against a fixed rubric.


## Study context
{task}


## Rubric (the exact CSV the grader uploaded)
```csv
{rubric_csv}
```


## Grading rules
- Score every scoreable rubric item (row) from 0 to that item's maximum points,
  reading every column of the CSV — point values and any full/partial/low-credit
  guidance — as written.
- Judge only what is present in the notebook transcript. Do not give credit for
  things the participant might have said aloud or done elsewhere.
- Use each item's scoring guidance on a consistent scale: 0 points when the
  criterion is missing, unattempted, or completely incorrect; roughly half of the
  maximum when the partial-credit guidance describes the notebook; the maximum
  only when the full-credit guidance is clearly met. Interpolate between anchors
  for in-between cases.
- Where an item's description says some evidence is scored under a different
  item, do NOT double-credit that evidence here.
- Cell outputs are truncated and figures are replaced by placeholders like
  "[image/png output: ...]"; treat such placeholders as evidence that the code
  ran and produced a figure, and judge the figure's purpose from the code.
- An empty or near-empty notebook should receive 0s with a brief explanation.
- Be consistent: the same evidence must always earn the same score.
- Reasoning: 1-3 concrete sentences referencing cell numbers.

## Output format
Respond with ONLY a JSON object (no markdown code fences, no commentary), exactly
this shape, one entry per scoreable rubric item, in the order they appear in the
CSV. Copy each item's section name and maximum points from the CSV verbatim:
{{"items": [{{"section": "<section from the CSV>", "criterion": "<short name of the \
rubric item from the CSV>", "max_pts": <the item's max points from the CSV>, "score": \
<number between 0 and max_pts>, "reasoning": "<1-3 sentences>"}}]}}"""


class GradedItem(BaseModel):
    section: str = ""
    criterion: str
    max_pts: float
    score: float
    reasoning: str = ""


class GradeResult(BaseModel):
    items: list[GradedItem]


def grade_notebook(
    client: Anthropic, task: str, rubric_csv: str, notebook_text: str
) -> list[dict]:
    """Score one notebook against the rubric CSV. The CSV text is given to the model
    exactly as uploaded; the model reads it and returns one score + reasoning per
    rubric item. Returns [{"section","criterion","max_pts","score","reasoning"}]."""
    user_content = (
        "Grade the following notebook.\n\n"
        f"===== BEGIN NOTEBOOK TRANSCRIPT =====\n{notebook_text}\n"
        "===== END NOTEBOOK TRANSCRIPT ====="
    )
    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            # Explicit timeout: without it the SDK refuses a non-streaming call whose
            # max_tokens *could* run >10 min. Grading takes well under that.
            timeout=600.0,
            system=[{
                "type": "text",
                "text": build_system_prompt(task, rubric_csv),
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_content}],
            output_format=GradeResult,
        )
    except ValidationError as e:
        raise RuntimeError(
            f"The model returned a malformed grading response: {e} — try again."
        ) from e
    if response.stop_reason == "refusal":
        detail = getattr(response.stop_details, "explanation", None) or "safety refusal"
        raise RuntimeError(f"The model declined to grade this notebook ({detail}).")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("The model ran out of room before finishing — try again.")
    parsed = response.parsed_output
    if parsed is None or not parsed.items:
        raise RuntimeError("The model returned no graded items — try again.")

    results = []
    for g in parsed.items:
        max_pts = max(0.0, float(g.max_pts))
        score = max(0.0, min(float(g.score), max_pts))  # clamp into [0, max]
        results.append(
            {
                "section": g.section.strip() or "General",
                "criterion": g.criterion.strip(),
                "max_pts": max_pts,
                "score": score,
                "reasoning": g.reasoning,
            }
        )
    return results


def call_with_errors_surfaced(fn, *args, **kwargs):
    """Run an API-calling function, converting SDK errors to readable messages."""
    try:
        return fn(*args, **kwargs)
    except anthropic.AuthenticationError:
        st.error(
            "Authentication failed. Set the ANTHROPIC_API_KEY environment variable "
            "(e.g. in a local .env file) and restart the app."
        )
    except anthropic.RateLimitError:
        st.error("Rate limited (or out of quota) — check your Anthropic plan/billing and try again.")
    except anthropic.APIStatusError as e:
        st.error(f"API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        st.error("Could not reach the Anthropic API — check your network connection.")
    except RuntimeError as e:
        st.error(str(e))
    return None


# ---------------------------------------------------------------------------
# Results tables + Excel export (built from the model's returned items)
# ---------------------------------------------------------------------------

def _section_order(graded: dict) -> list[str]:
    """Sections in first-seen order across all graded notebooks."""
    seen = []
    for rec in graded.values():
        for r in rec["results"]:
            if r["section"] not in seen:
                seen.append(r["section"])
    return seen


def _section_max(graded: dict, sections: list[str]) -> dict:
    """Max points per section — summed per notebook, then the max across notebooks."""
    section_max = {s: 0.0 for s in sections}
    for rec in graded.values():
        per_nb = {s: 0.0 for s in sections}
        for r in rec["results"]:
            per_nb[r["section"]] = per_nb.get(r["section"], 0.0) + float(r["max_pts"])
        for s in sections:
            section_max[s] = max(section_max[s], per_nb.get(s, 0.0))
    return section_max


def build_summary_df(graded: dict) -> pd.DataFrame:
    """One row per notebook: per-section totals, grand total, max, percent."""
    sections = _section_order(graded)
    section_max = _section_max(graded, sections)
    total_max = sum(section_max.values())

    rows = []
    for fname, rec in graded.items():
        section_score = {s: 0.0 for s in sections}
        for r in rec["results"]:
            section_score[r["section"]] = section_score.get(r["section"], 0.0) + float(r["score"])
        row = {"notebook": fname}
        for s in sections:
            row[f"{s} (/{section_max[s]:g})"] = round(section_score.get(s, 0.0), 2)
        total = round(sum(section_score.values()), 2)
        row["Total"] = total
        row["Max"] = round(total_max, 2)
        row["Percent"] = round(100 * total / total_max, 1) if total_max else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_details_df(graded: dict) -> pd.DataFrame:
    """One row per (notebook, rubric item), with score and reasoning."""
    rows = []
    for fname, rec in graded.items():
        for r in rec["results"]:
            rows.append(
                {
                    "notebook": fname,
                    "section": r["section"],
                    "criterion": r["criterion"],
                    "score": r["score"],
                    "max_pts": r["max_pts"],
                    "reasoning": r["reasoning"],
                }
            )
    return pd.DataFrame(rows)


def build_excel(graded: dict) -> bytes:
    """A workbook with a Summary sheet and a Details sheet."""
    summary = build_summary_df(graded)
    details = build_details_df(graded)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        details.to_excel(writer, sheet_name="Details", index=False)
    return buf.getvalue()
