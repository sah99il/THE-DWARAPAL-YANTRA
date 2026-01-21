import streamlit as st

def show_scores(identity_score, liveness_score):
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Identity Score", f"{identity_score:.3f}")

    with col2:
        st.metric("Liveness Score", f"{liveness_score:.3f}")
