import streamlit as st

def show_status(decision):
    if decision == "ACCEPT":
        st.success("✅ ACCESS GRANTED")
    else:
        st.error("❌ ACCESS DENIED")
