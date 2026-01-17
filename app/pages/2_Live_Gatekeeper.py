import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import streamlit as st
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from services.verification_service import verify_frame

st.title("🛡️ Step 2: Live Verification")

st.markdown("""
📷 Look at the camera and **hold still for a few seconds**.  
The system will automatically verify you.
""")

class HumanGatekeeper(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        result = verify_frame(img)

        # Friendly messages
        if result["verdict"]:
            text = f"✅ ACCESS GRANTED — Welcome {result['user']}"
            color = (0, 200, 0)
        else:
            reason = result.get("reason", "Analysing…")
            text = f"⏳ {reason}" if "Collecting" in reason else f"❌ ACCESS DENIED"
            color = (0, 0, 255)

        cv2.putText(
            img, text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
        )

        # Optional transparency toggle
        if "identity_score" in result:
            cv2.putText(
                img,
                f"Confidence: ID={result['identity_score']:.2f} | LIVE={result['liveness_score']:.2f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="human_gatekeeper",
    video_processor_factory=HumanGatekeeper,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)