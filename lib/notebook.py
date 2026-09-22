"""Flatten a Jupyter notebook into a plain-text transcript, or render it as HTML.

Shared by the quiz (question generation) and the grader (rubric scoring), which
previously each carried their own identical copy of this function.
"""

import re

import nbformat
from nbconvert import HTMLExporter

MAX_OUTPUT_CHARS_PER_CELL = 1500

_CELL_HEADER = re.compile(r"^--- (?:markdown|code) cell \d+ ---$|^--- output of cell \d+ ---$", re.M)


def notebook_has_content(notebook_text: str | None) -> bool:
    """True if the flattened notebook holds anything beyond the `--- code cell N ---`
    headers. A notebook of blank cells still flattens to a non-empty string, so a
    plain `.strip()` check lets it through."""
    return bool(notebook_text and _CELL_HEADER.sub("", notebook_text).strip())


def notebook_to_text(raw: bytes) -> str:
    """Flatten a .ipynb into a text transcript of markdown, code, and outputs.

    Markdown and code are never truncated; each cell's combined output text is
    capped at MAX_OUTPUT_CHARS_PER_CELL (marked "[output truncated]").
    """
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


def notebook_to_html(raw: bytes) -> str:
    """Render a .ipynb as a single self-contained HTML page, for the participant
    to view (a new browser tab, not the app itself) while debugging it.

    A small "Cell N" label is inserted before every cell, using the same 0-based,
    every-cell-type numbering as notebook_to_text's `--- ... cell N ---` markers --
    so a cell number in a write-up or an answer key points at the same cell here
    as it does in grading.
    """
    nb = nbformat.reads(raw.decode("utf-8"), as_version=4)
    labeled = []
    for i, cell in enumerate(nb.cells):
        label = nbformat.v4.new_markdown_cell(f"**Cell {i}**")
        label.metadata["tags"] = ["cell-label"]
        labeled.append(label)
        labeled.append(cell)
    nb.cells = labeled
    exporter = HTMLExporter(template_name="classic")
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(nb)
    return body
