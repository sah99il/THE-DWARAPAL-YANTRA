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

🗂️ Dataset Setup (MANDATORY)
⚠️ Datasets are NOT included in the repository.

Each collaborator must manually download and place datasets.

1️⃣ Identity Dataset (Face Recognition)

Recommended:

VGGFace2 (near fool, ~2 GB)

Folder structure:

data/identity/
├── n000001/
│   ├── img1.jpg
│   ├── img2.jpg
├── n000002/
│   ├── img1.jpg


Each folder = one identity.

2️⃣ Liveness Dataset (Spoof Detection)

Recommended:

CASIA-FASD (immada, ~2 GB)

Mapped structure:

data/liveness/
├── live/
│   ├── video1.mp4
│   ├── video2.mp4
├── spoof/
│   ├── print_attack.mp4
│   ├── replay_attack.mp4


📌 Note:
These videos are used offline for calibration and evaluation, not at runtime.

⚙️ Installation & Setup (CLONE & RUN)
1️⃣ Clone the repository
git clone <repo-url>
cd THE-DWARAPAL-YANTRA

2️⃣ Create virtual environment
python -m venv .venv
source .venv/bin/activate     # Linux/Mac
.venv\Scripts\activate        # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Place datasets

Manually place datasets in:

data/identity/
data/liveness/


Ensure data/ is ignored by Git.

5️⃣ Bulk enroll identities (ONE TIME)
python core/identity/bulk_enroll.py


This generates:

data/identity_db/
├── embeddings.npy
└── labels.json

6️⃣ Run the UI
streamlit run ui/app.py

🧪 Runtime Behavior (What to Expect)

When verification starts:

System collects frames
→ COLLECTING_FRAMES

Enforces minimum time window
→ WAITING_TIME

Accumulates liveness evidence
→ LIVE_CONFIRMED / UNSTABLE_SIGNAL

Final decision
→ ACCEPT or REJECT

Instant decisions are explicitly prevented.

🔐 Identity Handling

Identity verification is database-based

No session-only enrollment

Names are resolved from stored embeddings

System works across restarts

🛡️ Liveness Design (Important)

Uses temporal analysis, not single-frame inference

Uses buffered frames

Applies variance-based stability checks

Thresholds are dataset-calibrated

This design is explainable and PS-aligned.

🚫 What Is Intentionally NOT Done (Yet)

Training a deep CNN liveness model

Using datasets at runtime

Automatic Kaggle API downloads

These are deliberate design decisions, not omissions.


