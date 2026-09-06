"""Multipage entry point for the participant study platform.

Run with: streamlit run app.py

  Participant — the participant-facing intake: subject ID + condition, upload the
                7 study files, take the comprehension quiz. No scores are shown.
  Admin       — password-gated researcher view: per-participant review + cohort
                statistics + grading controls.
"""

import streamlit as st

st.set_page_config(page_title="Dynamic Evaluation Study", page_icon="🧪", layout="wide")

page = st.navigation([
    st.Page("app_pages/participant.py", title="Participant", icon=":material/assignment:"),
    st.Page("app_pages/admin.py", title="Admin", icon=":material/admin_panel_settings:"),
])

page.run()
