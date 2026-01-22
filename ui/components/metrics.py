import streamlit as st


def show_scores(identity_score, liveness_score):
    col1, col2 = st.columns(2)

    with col1:
        if identity_score is None:
            st.metric("Identity Score", "—")
        else:
            st.metric("Identity Score", f"{identity_score:.3f}")

    with col2:
        if liveness_score is None:
            st.metric("Liveness Score", "Collecting…")
        else:
            st.metric("Liveness Score", f"{liveness_score:.3f}")
