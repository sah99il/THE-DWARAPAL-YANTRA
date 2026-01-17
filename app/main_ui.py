import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import streamlit as st

st.set_page_config(
    page_title="Dwarapala — Identity & Liveness Verification",
    layout="centered"
)

st.title("👁️ Dwarapala")

st.markdown("""
### A Secure Identity Gatekeeper  

This system verifies:
- **Who you are** (Identity)
- **That you are real** (Liveness)

➡️ Use the sidebar to begin.
""")

st.info("Please ensure good lighting and face the camera directly.")