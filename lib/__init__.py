"""Shared modules for the participant study platform.

The three original tools (Notebook Quiz, Notebook Grader, Session Timeline) live
here as importable modules with no Streamlit page shell of their own:

  notebook  — flatten a .ipynb into a text transcript
  quiz      — quiz generation, self-check, and dynamic follow-ups (Claude Opus 5)
  quiz_ui   — the in-app quiz-taking flow and the read-only breakdown renderer
  grading   — grade one notebook against a rubric CSV (Claude Opus 5) + result tables
  timeline  — parse a Claude Code session .jsonl, chart it, and derive log metrics
  cohort    — per-participant dataframe and cross-condition statistics
"""
