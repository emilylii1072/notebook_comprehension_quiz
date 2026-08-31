"""Notebook Grader — grade many Jupyter notebooks against a rubric with Claude Opus 5.

Workflow (three tabs):
  1. Rubric & Task — upload a rubric CSV (or load a stored one) and review/edit the
     task description. The rubric CSV is handed to the model VERBATIM (never reparsed
     or transformed), so every column of grading guidance you wrote is used as-is.
  2. Grade notebooks — upload one or many `.ipynb` files; each is flattened to a
     transcript and sent to the Anthropic API, which reads the CSV rubric and returns
     a score + reasoning for every rubric item it finds. Results are saved per-notebook.
  3. Results — a summary table (section totals per notebook) and a details table
     (every rubric item + reasoning), downloadable as a multi-sheet Excel workbook.

One page of the multipage app — run the app via `streamlit run app.py`.
Auth: set ANTHROPIC_API_KEY (e.g. in a local .env file). Supabase is optional — without
it the tool still grades and exports; it just won't persist across sessions.
"""

import io

import anthropic
import nbformat
import pandas as pd
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from build_report import records_from_graded, render_report
from db import (
    delete_all_graded_notebooks,
    delete_graded_notebook,
    get_graded_notebooks,
    get_rubric,
    list_rubrics,
    save_graded_notebook,
    save_rubric,
    supabase_status,
    test_connection,
)

load_dotenv()

MODEL = "claude-opus-5"  # swap to "claude-sonnet-5" for cheaper/faster test iteration
MAX_TOKENS = 32000
MAX_OUTPUT_CHARS_PER_CELL = 1500
DEFAULT_RUBRIC_NAME = "attrition_v1"

# The default task graded against — the employee-attrition take-home. Editable in the
# UI, stored to Supabase alongside the rubric so every notebook is graded against the
# exact task text that was in effect.
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


# ---------------------------------------------------------------------------
# Notebook parsing
# ---------------------------------------------------------------------------

def notebook_to_text(raw: bytes) -> str:
    """Flatten a .ipynb into a text transcript of markdown, code, and outputs."""
    nb = nbformat.reads(raw.decode("utf-8"), as_version=4)
    parts = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown":
            parts.append(f"--- markdown cell {i} ---\n{cell.source}")
        elif cell.cell_type == "code":
            parts.append(f"--- code cell {i} ---\n{cell.source}")
            out_texts = []
            for out in cell.get("outputs", []):
                text = ""
                if out.get("output_type") == "stream":
                    text = "".join(out.get("text", ""))
                elif out.get("output_type") in ("execute_result", "display_data"):
                    text = "".join(out.get("data", {}).get("text/plain", ""))
                elif out.get("output_type") == "error":
                    text = "\n".join(out.get("traceback", []))
                if text:
                    if len(text) > MAX_OUTPUT_CHARS_PER_CELL:
                        text = text[:MAX_OUTPUT_CHARS_PER_CELL] + "\n[output truncated]"
                    out_texts.append(text)
            if out_texts:
                parts.append(f"--- output of cell {i} ---\n" + "\n".join(out_texts))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Grading (model call) — the rubric CSV is passed through verbatim, not parsed.
# ---------------------------------------------------------------------------

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
    # The system prompt (study context + rubric CSV + rules) is identical for every
    # notebook in a batch, so a cache_control breakpoint on it lets Anthropic prompt
    # caching serve it cheaply across the batch.
    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=MAX_TOKENS,
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
    """Max points per section — summed per notebook, then the max across notebooks
    (robust if one notebook's grading happened to drop or add an item)."""
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


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------

@st.cache_resource
def get_client() -> Anthropic:
    return Anthropic()


def _load_graded_from_db(rubric_name: str) -> dict:
    """Return {filename: {"results", "total_score", "max_score"}} from the DB."""
    graded = {}
    for row in get_graded_notebooks(rubric_name):
        graded[row["notebook_filename"]] = {
            "results": row["results"],
            "total_score": row["total_score"],
            "max_score": row["max_score"],
        }
    return graded


st.title("🎯 Notebook Grader")

if "grader_rubric" not in st.session_state:
    st.session_state.grader_rubric = None  # {"name","task","csv"}
if "graded" not in st.session_state:
    st.session_state.graded = {}  # filename -> {"results","total_score","max_score"}

tab_setup, tab_grade, tab_results, tab_report = st.tabs(
    ["1 · Rubric & Task", "2 · Grade notebooks", "3 · Results", "4 · Report"]
)

# ---- Tab 1: Rubric & Task ------------------------------------------------
with tab_setup:
    st.markdown(
        "Define the rubric and task once, then grade as many notebooks against it as "
        "you like. **Your rubric CSV is passed to the grader exactly as uploaded** — "
        "every column you wrote is used as-is. You can save it to the database and "
        "reload it later."
    )

    existing = list_rubrics()
    if existing:
        pick = st.selectbox(
            "Load a saved rubric", ["— new / upload below —"] + existing, index=0
        )
        if pick != "— new / upload below —" and st.button("Load this rubric"):
            r = get_rubric(pick)
            if r:
                st.session_state.grader_rubric = {
                    "name": r["name"],
                    "task": r["task"],
                    "csv": r["rubric_csv"],
                }
                st.session_state.graded = _load_graded_from_db(r["name"])
                st.success(f"Loaded rubric '{r['name']}'.")
                st.rerun()

    st.divider()
    rubric_name = st.text_input("Rubric name", value=DEFAULT_RUBRIC_NAME)
    uploaded_rubric = st.file_uploader("Rubric CSV", type=["csv"])
    task_text = st.text_area("Task description (graded against)", value=DEFAULT_TASK, height=260)

    if uploaded_rubric is not None and st.button("Save rubric", type="primary"):
        try:
            csv_text = uploaded_rubric.getvalue().decode("utf-8-sig")
        except UnicodeDecodeError:
            csv_text = uploaded_rubric.getvalue().decode("latin-1")
        name = rubric_name.strip() or DEFAULT_RUBRIC_NAME
        st.session_state.grader_rubric = {"name": name, "task": task_text, "csv": csv_text}
        ok, err = save_rubric(name, task_text, csv_text)
        if ok:
            st.success(f"Saved rubric '{name}' to database.")
        else:
            st.warning(f"Rubric ready for this session. Not saved to database: {err}")

    if st.session_state.grader_rubric:
        st.markdown("#### Active rubric")
        st.caption(f"**{st.session_state.grader_rubric['name']}**")
        csv_text = st.session_state.grader_rubric["csv"]
        try:
            st.dataframe(
                pd.read_csv(io.StringIO(csv_text)),
                use_container_width=True,
                hide_index=True,
            )
        except Exception:
            st.code(csv_text)  # preview only — the model always receives the raw CSV
        with st.expander("Raw CSV (exactly what the grader receives)"):
            st.code(csv_text, language="csv")

    st.divider()
    with st.expander("Database status (troubleshooting)"):
        status = supabase_status()
        if status["client_created"]:
            st.success("Supabase credentials found — results will persist.")
        else:
            st.warning(
                "No Supabase client. Grading and Excel export still work, but nothing "
                "persists. Add `SUPABASE_URL` / `SUPABASE_KEY` as **top-level** keys in "
                "Streamlit Cloud → Settings → Secrets (not nested under a `[section]` "
                "header), then **Reboot app**."
            )
        st.dataframe(
            pd.DataFrame(
                [
                    {"secret": name, **{k: str(v) for k, v in status[name].items()}}
                    for name in ("SUPABASE_URL", "SUPABASE_KEY", "ANTHROPIC_API_KEY")
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Top-level keys visible to the app: "
            + (", ".join(f"`{k}`" for k in status["top_level_secrets"]) or "_none_")
            + (f"  \nst.secrets error: {status['secrets_error']}"
               if status["secrets_error"] else "")
        )
        if st.button("Test database connection"):
            ok, msg = test_connection()
            (st.success if ok else st.error)(msg)

# ---- Tab 2: Grade notebooks ----------------------------------------------
with tab_grade:
    if not st.session_state.grader_rubric:
        st.info("Set up a rubric in the **Rubric & Task** tab first.")
    else:
        rub = st.session_state.grader_rubric
        st.markdown(
            f"Grading against **{rub['name']}**. Upload one or more notebooks — each is "
            "graded with a separate API call. Re-uploading a filename re-grades it."
        )
        uploaded_nbs = st.file_uploader(
            "Notebooks (.ipynb)", type=["ipynb"], accept_multiple_files=True
        )

        if uploaded_nbs and st.button(f"Grade {len(uploaded_nbs)} notebook(s)", type="primary"):
            progress = st.progress(0.0, text="Starting…")
            n = len(uploaded_nbs)
            for i, up in enumerate(uploaded_nbs):
                progress.progress(i / n, text=f"Grading {up.name} ({i + 1}/{n})…")
                try:
                    nb_text = notebook_to_text(up.getvalue())
                except Exception as e:
                    st.error(f"{up.name}: could not parse notebook — {e}")
                    continue
                if not nb_text.strip():
                    st.error(f"{up.name}: notebook appears to be empty.")
                    continue
                results = call_with_errors_surfaced(
                    grade_notebook, get_client(), rub["task"], rub["csv"], nb_text
                )
                if results is None:
                    continue  # error already surfaced
                total_score = round(sum(r["score"] for r in results), 2)
                max_score = round(sum(r["max_pts"] for r in results), 2)
                st.session_state.graded[up.name] = {
                    "results": results,
                    "total_score": total_score,
                    "max_score": max_score,
                }
                ok, err = save_graded_notebook(
                    {
                        "rubric_name": rub["name"],
                        "notebook_filename": up.name,
                        "notebook_text": nb_text,
                        "results": results,
                        "total_score": total_score,
                        "max_score": max_score,
                    }
                )
                note = "" if ok else f"  ⚠️ not saved to DB ({err})"
                st.write(f"✅ {up.name}: {total_score:g} / {max_score:g}{note}")
            progress.progress(1.0, text="Done.")

        if st.session_state.graded:
            st.caption(f"{len(st.session_state.graded)} notebook(s) graded this session.")

# ---- Tab 3: Results ------------------------------------------------------
with tab_results:
    if not st.session_state.grader_rubric:
        st.info("Set up a rubric and grade some notebooks first.")
    elif not st.session_state.graded:
        st.info("No graded notebooks yet — grade some in the **Grade notebooks** tab.")
    else:
        graded = st.session_state.graded
        rubric_name = st.session_state.grader_rubric["name"]

        st.markdown("#### Summary")
        st.dataframe(build_summary_df(graded), use_container_width=True, hide_index=True)

        st.markdown("#### Details (per rubric item)")
        st.dataframe(build_details_df(graded), use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download Excel (Summary + Details)",
            data=build_excel(graded),
            file_name=f"grades_{rubric_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )

        st.divider()
        st.markdown("#### Remove saved notebooks")
        to_remove = st.multiselect(
            "Select graded notebooks to remove", sorted(graded.keys())
        )
        col_rm, col_clear = st.columns(2)
        if col_rm.button("Remove selected", disabled=not to_remove):
            errors = []
            for fname in to_remove:
                ok, err = delete_graded_notebook(rubric_name, fname)
                if ok:
                    st.session_state.graded.pop(fname, None)
                else:
                    errors.append(f"{fname}: {err}")
            if errors:
                st.warning("Some deletions failed in the database:\n\n" + "\n".join(errors))
            else:
                st.success(f"Removed {len(to_remove)} notebook(s).")
            st.rerun()
        if col_clear.button(f"Clear ALL for '{rubric_name}'", type="secondary"):
            ok, err = delete_all_graded_notebooks(rubric_name)
            st.session_state.graded = {}
            if ok:
                st.success("Cleared all graded notebooks for this rubric.")
            else:
                st.warning(f"Cleared this session, but the database delete failed: {err}")
            st.rerun()

# ---- Tab 4: Report -------------------------------------------------------
with tab_report:
    if not st.session_state.grader_rubric:
        st.info("Set up a rubric and grade some notebooks first.")
    elif not st.session_state.graded:
        st.info("No graded notebooks yet — grade some in the **Grade notebooks** tab.")
    else:
        rubric_name = st.session_state.grader_rubric["name"]
        st.markdown(
            "Visual review of every notebook graded against this rubric: score "
            "distribution, per-section spread, a notebook × criterion heatmap, the "
            "criteria the cohort did worst on, and a clustering of scoring profiles. "
            "Hover any mark for detail; click one to jump to that notebook."
        )
        records = records_from_graded(st.session_state.graded)
        report_html = render_report(records, rubric_name)
        st.download_button(
            "⬇️ Download report (single HTML file)",
            data=report_html,
            file_name=f"report_{rubric_name}.html",
            mime="text/html",
        )
        # An iframe, not st.html: the report ships its own CSS reset and tooltip
        # script, which must not leak into (or inherit from) the app's styles.
        # height="content" lets the page scroll normally instead of nesting a
        # scrollbar inside a fixed frame.
        st.iframe(report_html, height="content")
