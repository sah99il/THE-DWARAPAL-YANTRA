DWARAPALA – Live Biometric Gatekeeper

Dwarapala is a real-time biometric verification system that performs face recognition and liveness detection together using a live webcam feed.

The system is designed to verify two things at the same time:

whether the person matches an enrolled identity

whether the person is physically present and not a spoof

The project focuses on practical security issues that appear in real deployments, not just lab-style face recognition accuracy.

Overview

Most face recognition systems only answer one question: who is this person?
Dwarapala also answers a second, equally important question: is this person real?

Instead of treating face recognition and liveness detection as separate modules, both are evaluated and fused before making a final decision. Access is granted only when identity and liveness are both confirmed.

Motivation

In real-world usage, many face recognition systems fail because they:

Accept printed photographs

Accept mobile screen replays

Perform poorly when matching ID photos with live camera images

Make decisions based on a single frame

Dwarapala addresses these problems by:

Using temporal information instead of single-frame decisions

Separating identity verification from liveness detection

Applying score-level fusion with conservative thresholds

System Pipeline
Webcam Input
    ↓
Face Detection and Alignment
    ↓
Face Embedding Extraction (ArcFace)
    ↓
Identity Matching (Cosine Similarity)
    ↓
Temporal Liveness Analysis
    ↓
Score-Level Fusion
    ↓
Final Decision (ALLOW / DENY)


Each stage is implemented as a separate module to keep the system modular and easy to inspect.

Core Components
Face Recognition (Identity Verification)

Model: ArcFace (InsightFace ONNX)

Input: Aligned 112×112 RGB face image

Output: 512-dimensional normalized embedding

Similarity metric: Cosine similarity

Acceptance threshold: 0.65

This setup provides reasonable robustness against lighting changes, aging effects, and low-quality or scanned ID photos.
User embeddings are stored in a local SQLite database during enrollment.

Liveness Detection

Liveness is evaluated over time rather than on individual frames.

The system observes short-term facial behavior using:

Motion consistency across frames

Eye blink detection

Texture variation between real skin and flat surfaces

Temporal window:

Approximately 20–30 frames (around 2 seconds)

This approach helps resist:

Printed photo attacks

Screen replay attacks

Static or looped spoofing attempts

No medical or unrealistic claims are made. All cues are proxy-based and explainable.

Score-Level Fusion

The final decision is made only when both conditions are satisfied:

identity_score ≥ 0.65
liveness_score ≥ 0.70


This fusion strategy intentionally prioritizes security over convenience and reduces false acceptances.

Performance Notes

Runtime latency: under 500 ms per frame on CPU

Designed for low false acceptance

Suitable for real-time use on consumer hardware

Evaluation metrics considered:

True Accept Rate at fixed False Accept Rate

Average Classification Error Rate

Inference latency
