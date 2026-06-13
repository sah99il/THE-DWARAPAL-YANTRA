import sys
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import torch
import cv2
import time

from core.decision.verifier import DwarapalVerifier
from face_detect import FaceDetector


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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    color: white;
}
#MainMenu, footer, header {visibility: hidden;}

/* Glassmorphism Card */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    margin-bottom: 24px;
}

h1 {
    font-weight: 800;
    background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0px;
}

.subtitle {
    text-align: center;
    color: #94A3B8;
    font-size: 1.1rem;
    margin-top: 8px;
    margin-bottom: 24px;
}

.stButton>button {
    width: 100%;
    height: 3.5em;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 600;
    border: none;
    background: linear-gradient(90deg, #3b82f6, #6366f1);
    color: white;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #2563eb, #4f46e5);
    transform: translateY(-2px);
    box-shadow: 0 10px 20px -10px rgba(99, 102, 241, 0.5);
}
</style>

<div class="glass-card">
    <h1>DWARAPAL YANTRA</h1>
    <div class="subtitle">Real-Time Facial Recognition System</div>
</div>
""", unsafe_allow_html=True)
# =====================================================


@st.cache_resource
def load_verifier():
    return DwarapalVerifier(
        config_path=os.path.join(_PROJECT_ROOT, "configs", "system.yaml"),
        identity_ckpt=os.path.join(_PROJECT_ROOT, "models", "checkpoints", "identity", "embedder_epoch_10.pth")
    )

@st.cache_resource
def load_detector():
    return FaceDetector(device="cpu")

verifier = load_verifier()
detector = load_detector()


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
        frame_slot.image(frame, channels="BGR", use_container_width=True)

        faces = detector.detect(frame)
        if faces:
            # Get largest face
            f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            x1, y1, x2, y2 = map(int, f.bbox)
            face_img = frame[max(0, y1):y2, max(0, x1):x2]
            
            if face_img.size > 0:
                st.session_state.face_frames.append({
                    "img": face_img,
                    "embedding": getattr(f, "embedding", None)
                })

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

    time.sleep(0.3)
    st.rerun()
# =======================================================


# ===================== SAVE ENROLL =====================
if st.session_state.mode == "save_enroll":
    st.subheader("Save New Identity")
    name = st.text_input("Enter user name")

    if st.button("Save") and name.strip():
        # Fallback to ViT model for enrollment since verifier.enroll_new_user 
        # is hardcoded to run inputs through the identity model
        faces = [preprocess_face(f["img"]) for f in st.session_state.face_frames]
        verifier.enroll_new_user(name.strip(), faces)
        st.session_state.result = f"User '{name}' enrolled successfully"
        st.session_state.mode = "idle"
        st.rerun()
# =======================================================


# ===================== FINAL VERIFY =====================
if st.session_state.mode == "finalize_verify":
    for f in st.session_state.face_frames:
        verifier.add_frame(f["img"])

    last_f = st.session_state.face_frames[-1]
    last_face_tensor = preprocess_face(last_f["img"])
    
    # Bypass random ViT checkpoint and use native InsightFace embedding for identity matching
    name, id_score = verifier.identify(last_face_tensor, face_embedding=last_f["embedding"])
    live_score, live_state = verifier.evaluate_liveness()
    result = verifier.verify(id_score, (live_score, live_state))

    decision = "ACCESS GRANTED" if result["decision"] == "ACCEPT" else "ACCESS DENIED"

    st.session_state.result = (
        f"Identity: {name} | {decision}"
    )

    st.session_state.mode = "idle"
    st.rerun()
# =======================================================
