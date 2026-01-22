# 👁️ DWARAPAL YANTRA
**A Temporal Biometric Gatekeeper with Identity Verification and Liveness Detection**

---

## 📌 Project Overview

**DWARAPAL YANTRA** is a real-time biometric verification system that combines:

- Database-based face identity recognition  
- Temporal liveness detection  
- Explainable decision logic  
- Live webcam verification  

The system is designed to address common biometric vulnerabilities such as:

- Replay attacks  
- Printed photo spoofing  
- Session-only identity checks  

By enforcing **temporal evidence accumulation** and **database-driven verification**.

---

## 🎯 Problem Statement Alignment

This project satisfies the following core requirements:

- Identity verification against an existing database  
- Liveness detection using **temporal cues** (not single-frame inference)  
- Explainable accept/reject decisions  
- Resistance to spoofing attacks  
- Separation of training, calibration, and runtime inference  

> ⚠️ The system explicitly avoids instant liveness decisions and enforces a minimum observation time, as required by the problem statement.

---

## 🧠 System Architecture

### High-level Flow

```text
Live Camera Frame
        ↓
Face Preprocessing (224×224)
        ↓
Identity Identification (Database-based)
        ↓
Temporal Frame Buffer
        ↓
Liveness Evaluation (Texture + Temporal + rPPG)
        ↓
Evidence Accumulation (Time + Stability)
        ↓
Final Decision (ACCEPT / REJECT)


---

## 📁 Project Structure

```text
THE-DWARAPAL-YANTRA/
├── core/
│   ├── identity/
│   │   ├── vit_embedder.py        # ViT-based face embedding model
│   │   ├── database.py            # Persistent identity database
│   │   └── bulk_enroll.py         # One-time dataset enrollment
│   ├── liveness/
│   │   ├── texture.py             # Texture-based cues
│   │   ├── temporal.py            # Temporal motion cues
│   │   ├── rppg.py                # rPPG-based cues
│   │   └── fusion.py              # Score fusion
│   └── decision/
│       └── verifier.py            # Unified verification engine
│
├── ui/
│   ├── app.py                     # Streamlit UI
│   ├── components/               # UI components
│   └── visualizers/              # Gauges & indicators
│
├── data/
│   ├── identity/                 # Face images (per person folder)
│   ├── identity_db/              # Generated embeddings + labels (ignored)
│   └── liveness/                 # Live & spoof videos (calibration only)
│
├── training/                     # Model training scripts (offline)
├── configs/
│   └── system.yaml               # System thresholds & parameters
├── requirements.txt
├── .gitignore
└── README.md
