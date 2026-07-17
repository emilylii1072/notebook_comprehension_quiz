"""Multipage entry point.

Run with: streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="Dynamic Evaluation Tools", page_icon="🧰", layout="centered")

page = st.navigation([
    st.Page("app_pages/notebook_quiz.py", title="Notebook Quiz", icon=":material/quiz:"),
    st.Page("app_pages/session_timeline.py", title="Session Timeline", icon=":material/timeline:"),
])

page.run()
