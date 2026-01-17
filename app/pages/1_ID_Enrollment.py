import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import streamlit as st
import cv2
import numpy as np
from services.id_enrollment_service import enroll_from_id_image

st.title("🪪 Step 1: Enroll Identity")

st.markdown("""
Upload a **government ID photo**.
Make sure the face is **clear and front-facing**.
""")

name = st.text_input("Your Name")

uploaded = st.file_uploader(
    "Upload ID Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    image = cv2.imdecode(
        np.frombuffer(uploaded.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    st.image(image, caption="Uploaded ID Image", channels="BGR")

    if st.button("Enroll Identity"):
        with st.spinner("Extracting identity…"):
            result = enroll_from_id_image(image, name)

        if result["success"]:
            st.success("✅ Identity enrolled successfully!")
        else:
            st.error(f"❌ {result['reason']}")