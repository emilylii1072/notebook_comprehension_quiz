"""One shared, cached Anthropic client for every page and module.

`Anthropic()` reads ANTHROPIC_API_KEY from the environment (populated from a local
`.env` via python-dotenv, or the deployment's secrets manager).
"""

import streamlit as st
from anthropic import Anthropic


@st.cache_resource
def get_client() -> Anthropic:
    return Anthropic()
