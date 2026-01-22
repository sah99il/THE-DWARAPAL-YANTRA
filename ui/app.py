import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import torch
import cv2
import time

from ui.components.camera import Camera
from ui.components.metrics import show_scores
from ui.components.status import show_status
from ui.visualizers.gauges import gauge
from core.decision.verifier import DwarapalVerifier


# ------------------------
# Face preprocessing
# ------------------------
def preprocess_face(frame_rgb, size=224):
    face = cv2.resize(frame_rgb, (size, size))
    face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float()
    return face / 255.0


# ------------------------
# Streamlit setup
# ------------------------
st.set_page_config(page_title="DWARAPAL YANTRA", layout="wide")
st.title("👁️ DWARAPAL — Live Biometric Gatekeeper")


@st.cache_resource
def load_verifier():
    return DwarapalVerifier(
        config_path="configs/system.yaml",
        identity_ckpt="models/checkpoints/identity/embedder_epoch_10.pth"
    )


verifier = load_verifier()
camera = Camera()
frame_placeholder = st.empty()


# ------------------------
# Session state
# ------------------------
if "running" not in st.session_state:
    st.session_state.running = False


# ------------------------
# Controls
# ------------------------
st.sidebar.header("Live Verification")

if st.sidebar.button("▶️ Start"):
    st.session_state.running = True
    verifier.buffer.clear()
    verifier.live_scores.clear()
    verifier.start_time = None

if st.sidebar.button("⏹ Stop"):
    st.session_state.running = False
    verifier.buffer.clear()
    verifier.live_scores.clear()
    verifier.start_time = None


# ------------------------
# Main real-time loop
# ------------------------
if st.session_state.running:
    frame = camera.read()

    if frame is not None:
        frame_placeholder.image(frame)

        # ---------- Identity ----------
        face_tensor = preprocess_face(frame)
        name, id_score = verifier.identify(face_tensor)

        # ---------- Liveness ----------
        verifier.add_frame(frame)
        live_score, live_state = verifier.evaluate_liveness()

        # ---------- Decision ----------
        result = verifier.verify(id_score, (live_score, live_state))

        # ---------- UI ----------
        st.subheader(f"Identity: **{name}**")

        show_scores(
            result["identity_score"],
            result["liveness_score"]
        )

        gauge("Identity Confidence", result["identity_score"])
        gauge("Liveness Confidence", result["liveness_score"])

        # Explainable status (PS-aligned)
        show_status(result["decision"])

    time.sleep(0.03)
