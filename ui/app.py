import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import torch
import numpy as np
import cv2

from ui.components.camera import Camera
from ui.components.metrics import show_scores
from ui.components.status import show_status
from ui.visualizers.gauges import gauge
from core.decision.verifier import DwarapalVerifier

# ------------------------
# Face preprocessing (MANDATORY)
# ------------------------
def preprocess_face(frame_rgb, size=224):
    """
    frame_rgb: (H, W, 3) uint8
    returns: (1, 3, 224, 224) float tensor
    """
    face = cv2.resize(frame_rgb, (size, size))
    face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float()
    face = face / 255.0
    return face


# ------------------------
# UI setup
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

# ------------------------
# Session state
# ------------------------
if "enrolled" not in st.session_state:
    st.session_state.enrolled = False

if "running" not in st.session_state:
    st.session_state.running = False

frame_placeholder = st.empty()

# ------------------------
# Enrollment (FIXED)
# ------------------------
st.sidebar.header("Step 1: Identity Enrollment")

if st.sidebar.button("📸 Capture ID Face"):
    frame = camera.read()
    if frame is not None:
        face_tensor = preprocess_face(frame)   # ✅ FIX
        verifier.enroll_identity(face_tensor)
        st.session_state.enrolled = True
        st.sidebar.success("Identity Enrolled")

# ------------------------
# Live Verification
# ------------------------
st.sidebar.header("Step 2: Live Verification")

if st.sidebar.button("▶️ Start Verification"):
    if not st.session_state.enrolled:
        st.sidebar.error("Enroll identity first")
    else:
        st.session_state.running = True

# ------------------------
# Main loop (FIXED)
# ------------------------
if st.session_state.running:
    frame = camera.read()
    if frame is not None:
        frame_placeholder.image(frame)

        verifier.add_frame(frame)
        st.session_state.frame_count += 1

        # Run liveness every 10 frames
        if (
            len(verifier.buffer) >= verifier.window_size and
            st.session_state.frame_count % 10 == 0
        ):
            face_tensor = preprocess_face(frame)
            result = verifier.verify(face_tensor)

            show_scores(result["identity_score"], result["liveness_score"])
            gauge("Identity Confidence", result["identity_score"])
            gauge("Liveness Confidence", result["liveness_score"])
            show_status(result["decision"])

# ------------------------
# Session state
# ------------------------
if "enrolled" not in st.session_state:
    st.session_state.enrolled = False

if "running" not in st.session_state:
    st.session_state.running = False

# ✅ ADD THIS
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
