# config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

ENROLLED_USERS_DIR = PROJECT_ROOT / "enrolled_users"
DB_PATH = ENROLLED_USERS_DIR / "faces.db"

SIMILARITY_THRESHOLD = 0.40
NUM_ENROLL_EMBEDDINGS = 5

# Face detection quality gates (helps large-scale accuracy)
MIN_DET_SCORE = 0.6
MIN_FACE_SIZE = 80  # min(width, height) in pixels

# Enrollment quality: collect more samples, keep best ones
ENROLL_SAMPLES = 20

# Verification stability / safety
EMB_SMOOTH_FRAMES = 5
SCORE_GAP_THRESHOLD = 0.05  # top1 - top2 must exceed this, else Unknown

# Liveness (spoof detection)
LIVENESS_ENABLED = True
# These heuristics vary by camera/lighting. We use slightly relaxed defaults
# plus multi-frame confirmation in verify.py to reduce false positives.
LIVENESS_LBP_THRESH = 0.006
LIVENESS_FFT_THRESH = 2.8
LIVENESS_FAIL_FRAMES = 5
LIVENESS_PASS_FRAMES = 3
LIVENESS_MODEL_PATH = ENROLLED_USERS_DIR / "liveness_model.pt"
LIVENESS_MODEL_SPOOF_THRESHOLD = 0.5

# UI / stability
NAME_STABLE_FRAMES = 3
