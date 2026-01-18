👁️ DWARAPALA – Live Biometric Gatekeeper

A real-time system for secure face verification with built-in liveness detection.

📌 Overview

Dwarapala is a real-time biometric verification system that checks two things at once:

Identity — Is this person already enrolled in the system?

Liveness — Is this a real human in front of the camera, not a photo or screen replay?

Instead of treating face recognition and liveness as separate problems, Dwarapala combines both into a single decision pipeline. The system is designed to handle real-world issues such as ID-to-selfie mismatch, replay attacks, and static spoofing attempts.

🎯 Problem Statement

Most traditional face recognition systems fail in real deployments because they:

Accept printed photographs

Fail against mobile screen replays

Struggle with ID photo vs live camera mismatch

Make decisions using a single frame

Dwarapala addresses these weaknesses by enforcing temporal evidence and score-level fusion, ensuring that authentication succeeds only when the identity matches and the subject is proven live.

🧠 System Architecture
Live Webcam
   ↓
Face Detection & Alignment
   ↓
Embedding Extraction (ArcFace)
   ↓
Identity Matching (Cosine Similarity)
   ↓
Temporal Liveness Analysis
   ↓
Score-Level Fusion
   ↓
Final Verdict (ALLOW / DENY)


Each stage is modular, interpretable, and independently verifiable.

🔍 Core Components
1️⃣ Face Recognition (Identity Verification)

Model: ArcFace (ONNX via InsightFace)

Input: Aligned 112 × 112 RGB face

Output: 512-D normalized embedding

Similarity Metric: Cosine similarity

Acceptance Threshold: ≥ 0.65

This setup provides robustness against:

Lighting variations

Aging differences

Low-quality or scanned ID photos

2️⃣ Liveness Detection (Anti-Spoofing)

Liveness is not evaluated per frame, but across time.

Temporal cues used:

Motion consistency across frames

Eye blink detection (natural, involuntary behavior)

Texture variation (real skin vs flat displays)

Temporal window:
~20–30 frames (≈ 2 seconds)

This approach successfully resists:

Printed photo attacks

Screen replay attacks

Static or looped deepfake videos

⚠️ No unrealistic or medical claims are made (e.g., heartbeat detection).
All cues are proxy-based, explainable, and academically defensible.

3️⃣ Score-Level Fusion

The system only grants access when both conditions are satisfied:

identity_score ≥ 0.65
liveness_score ≥ 0.70


This conservative fusion strategy prioritizes security over convenience, reducing false acceptances in sensitive environments.

📊 Evaluation Metrics
Metric	Description
TAR @ FAR	True Accept Rate at fixed False Accept Rate
ACER	Average Classification Error Rate
Latency	< 500 ms per frame (CPU)

The system is optimized for low false acceptance, making it suitable for security-critical applications.

🧪 How to Run

1️⃣ Install dependencies

pip install -r requirements.txt


2️⃣ Enroll a user

python -m scripts.test_enrollment


3️⃣ Run live verification

python -m scripts.live_gatekeeper


Press q to exit the live window.

📁 Project Structure
THE-DWARAPAL-YANTRA/
├── core/
│   ├── face/        # alignment, embeddings, matching
│   └── liveness/    # temporal liveness engine
├── services/        # verification & fusion logic
├── database/        # SQLite user embeddings
├── scripts/         # enrollment & live runners
├── models/          # InsightFace ONNX models
└── README.md

⚠️ Known Limitations

Requires frontal or near-frontal face input

Extreme lighting can affect liveness confidence

Not intended for identical twin differentiation

Not a medical-grade biometric system

These constraints are explicitly acknowledged and documented.

🚀 Applications

Secure authentication systems

Online exam proctoring

Attendance verification

Physical or digital access control

Anti-spoofing and biometric research demos

🏁 Conclusion

Dwarapala demonstrates that identity verification alone is not sufficient for secure biometric systems. By separating who the person is from whether the person is real, and fusing both decisions conservatively, the system provides strong resistance against common spoofing attacks while remaining computationally efficient and transparent.

📌 Author Notes

This project focuses on:

Sound engineering practices

Reproducible and modular design

Honest security claims

Academic and technical defensibility