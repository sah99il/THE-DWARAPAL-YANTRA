👁️ DWARAPALA – LIVE BIOMETRIC GATEKEEPER

A unified system for cross-domain face verification and real-time liveness detection.

📌 Overview

Dwarapala is a real-time biometric verification system designed to validate both:

Identity — Is this person present in the enrolled database?

Liveness — Is this a real human or a spoof (photo / replay / screen)?

The system bridges the gap between static identity documents and live camera input, addressing challenges such as domain shift, replay attacks, and temporal spoofing.

🎯 Problem Statement

Traditional face recognition systems are vulnerable to:

Printed photo attacks

Screen replay attacks

Domain mismatch (ID photo vs live selfie)

Single-frame spoofing

Dwarapala solves this by enforcing temporal liveness constraints and score-level fusion, ensuring that only a living, matching identity is accepted.

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

🔍 Core Components
1️⃣ Face Recognition (Identity)

Model: ArcFace (ONNX, InsightFace)

Input: Aligned 112×112 RGB face

Output: 512-D normalized embedding

Similarity Metric: Cosine similarity

Decision Threshold: ≥ 0.65

This allows robust matching even under:

Aging

Lighting changes

Low-quality ID images

2️⃣ Liveness Detection (Anti-Spoofing)

Liveness is evaluated temporally, not per-frame.

Signals Used:

Motion Consistency (frame-to-frame changes)

Eye Blink Detection (involuntary physiological cue)

Texture Variance (flat screens vs real skin)

Temporal Window:

~20–30 frames (~2 seconds)

This defeats:

Printed photos

Mobile screen replays

Static deepfake loops

⚠️ No false medical claims (e.g., heartbeat detection).
Liveness cues are proxy-based and defensible.

3️⃣ Score-Level Fusion

Final decision is made only if both checks pass:

identity_score ≥ 0.65
liveness_score ≥ 0.70


Fusion logic enforces a conservative security policy.

📊 Evaluation Metrics
Metric	Description
TAR @ FAR	True Accept Rate at fixed False Accept Rate
ACER	Average Classification Error Rate
Latency	< 500 ms per frame (CPU)

The system prioritizes low false acceptance, suitable for security-critical use cases.

🧪 How to Run
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Enroll a user
python -m scripts.test_enrollment

3️⃣ Run live verification (OpenCV window)
python -m scripts.live_gatekeeper


Press q to exit.

📁 Project Structure
THE-DWARAPAL-YANTRA/
├── core/
│   ├── face/           # alignment, embedding, matching
│   └── liveness/       # temporal liveness engine
├── services/           # verification & fusion logic
├── database/           # SQLite user embeddings
├── scripts/            # test & live runners
├── models/             # InsightFace ONNX models
└── README.md

⚠️ Limitations

Requires frontal or near-frontal face

Extreme lighting may reduce liveness confidence

Not designed for identical twin differentiation

Not a medical-grade biometric system

These limitations are explicitly acknowledged.

🚀 Applications

Secure authentication systems

Exam proctoring

Attendance verification

Access control systems

Anti-spoof research demos

🏁 Conclusion

Dwarapala demonstrates how identity verification and liveness detection must be treated as independent but fused security checks.
By enforcing temporal evidence, the system resists common spoofing attacks while remaining computationally efficient and interpretable.

📌 Author Notes

This project emphasizes:

Correct engineering practices

Reproducible environments

Honest security claims

Academic defensibility

