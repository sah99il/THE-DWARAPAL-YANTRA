import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import torch
import cv2
import time

from core.decision.verifier import DwarapalVerifier
from core.liveness.face_roi import extract_face_roi


# ===================== CONFIG =====================
ENROLL_SECONDS = 2.0
VERIFY_SECONDS = 4.0
CAMERA_INDEX = 0
# =================================================


# ===================== HELPERS =====================
def preprocess_face(face_rgb, size=224):
    face = cv2.resize(face_rgb, (size, size))
    face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float()
    return face / 255.0


def status_card(text, color="#020617"):
    st.markdown(
        f"""
        <div style="
            padding:16px;
            border-radius:12px;
            background-color:{color};
            text-align:center;
            font-size:18px;
            font-weight:600;
            border:1px solid #1E293B;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )
# =================================================


# ===================== PAGE SETUP =====================
st.set_page_config(page_title="DWARAPAL YANTRA", layout="centered")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
button {
    width: 100%;
    height: 3em;
    border-radius: 10px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style="text-align:center;">👁️ DWARAPAL</h1>
<p style="text-align:center; color:#94A3B8;">
Real-Time Identity & Liveness Verification System
</p>
<hr style="border:1px solid #1E293B;">
""", unsafe_allow_html=True)
# =====================================================


@st.cache_resource
def load_verifier():
    return DwarapalVerifier(
        config_path="configs/system.yaml",
        identity_ckpt="models/checkpoints/identity/embedder_epoch_10.pth"
    )


verifier = load_verifier()


# ===================== SESSION STATE =====================
if "mode" not in st.session_state:
    st.session_state.mode = "idle"  # idle | enroll | verify

if "start_time" not in st.session_state:
    st.session_state.start_time = None

if "face_frames" not in st.session_state:
    st.session_state.face_frames = []

if "result" not in st.session_state:
    st.session_state.result = None
# =======================================================


# ===================== IDLE SCREEN =====================
if st.session_state.mode == "idle":
    col1, col2 = st.columns(2)

    if col1.button("➕ Enroll New User"):
        st.session_state.mode = "enroll"
        st.session_state.start_time = time.time()
        st.session_state.face_frames = []
        verifier.reset_liveness()
        st.rerun()

    if col2.button("🔐 Verify Identity"):
        st.session_state.mode = "verify"
        st.session_state.start_time = time.time()
        st.session_state.face_frames = []
        verifier.reset_liveness()
        st.rerun()

    if st.session_state.result:
        st.divider()
        status_card(
            st.session_state.result,
            "#064E3B" if "GRANTED" in st.session_state.result else "#7F1D1D"
        )
# =======================================================


# ===================== CAMERA STATES =====================
if st.session_state.mode in ["enroll", "verify"]:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    frame_slot = st.empty()

    ret, frame = cap.read()
    cap.release()

    if ret:
        frame_slot.image(frame, channels="BGR", use_column_width=True)

        face = extract_face_roi(frame)
        if face is not None:
            st.session_state.face_frames.append(face)

    elapsed = time.time() - st.session_state.start_time

    if st.session_state.mode == "enroll":
        status_card("Recording enrollment clip…", "#020617")
        if elapsed >= ENROLL_SECONDS:
            st.session_state.mode = "save_enroll"
            st.rerun()

    if st.session_state.mode == "verify":
        status_card("Analyzing live video…", "#020617")
        if elapsed >= VERIFY_SECONDS:
            st.session_state.mode = "finalize_verify"
            st.rerun()

    time.sleep(0.05)
    st.rerun()
# =======================================================


# ===================== SAVE ENROLL =====================
if st.session_state.mode == "save_enroll":
    st.subheader("Save New Identity")
    name = st.text_input("Enter user name")

    if st.button("Save") and name.strip():
        faces = [preprocess_face(f) for f in st.session_state.face_frames]
        verifier.enroll_new_user(name.strip(), faces)
        st.session_state.result = f"User '{name}' enrolled successfully"
        st.session_state.mode = "idle"
        st.rerun()
# =======================================================


# ===================== FINAL VERIFY =====================
if st.session_state.mode == "finalize_verify":
    for f in st.session_state.face_frames:
        verifier.add_frame(f)

    last_face = preprocess_face(st.session_state.face_frames[-1])
    name, id_score = verifier.identify(last_face)
    live_score, live_state = verifier.evaluate_liveness()
    result = verifier.verify(id_score, (live_score, live_state))

    decision = "ACCESS GRANTED" if result["decision"] == "ACCEPT" else "ACCESS DENIED"

    st.session_state.result = (
        f"Identity: {name} | {decision}"
    )

    st.session_state.mode = "idle"
    st.rerun()
# =======================================================
