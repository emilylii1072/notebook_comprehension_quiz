"""One shared, cached Anthropic client for every page and module.

The key is read via `db.get_secret` (Streamlit Cloud's `st.secrets` first, then an
env var / local `.env`) and passed explicitly — a bare `Anthropic()` only checks
`os.environ`, which Streamlit Community Cloud does not reliably populate from its
secrets manager.
"""

import streamlit as st
from anthropic import Anthropic

from db import get_secret


@st.cache_resource
def get_client() -> Anthropic:
    key = get_secret("ANTHROPIC_API_KEY")
    return Anthropic(api_key=key) if key else Anthropic()
