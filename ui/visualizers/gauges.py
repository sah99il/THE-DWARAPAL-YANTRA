import streamlit as st

def gauge(label, value):
    st.progress(min(max(value, 0.0), 1.0), text=f"{label}: {value:.2f}")
