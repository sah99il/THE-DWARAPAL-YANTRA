from core.face.aligner import align_face
from core.face.extractor import FaceExtractor
from core.face.matcher import FaceMatcher
from core.liveness.liveness_engine import LivenessDetector
from database.db_manager import DatabaseManager
from services.fusion_service import fuse

# --------- Singletons (loaded once) ----------
face_extractor = FaceExtractor()
matcher = FaceMatcher(threshold=0.65)
liveness = LivenessDetector(window_size=30)   # ~1 second @ 30fps
db = DatabaseManager()

MIN_LIVE_FRAMES = 15  # required temporal evidence


def verify_frame(frame):
    """
    Verifies identity + liveness from a single frame in a stream.
    This function is meant to be called repeatedly over time.
    """

    # 0. Safety
    if frame is None:
        return {
            "verdict": False,
            "reason": "Empty frame"
        }

    # 1. Face alignment
    aligned = align_face(frame)
    if aligned is None:
        return {
            "verdict": False,
            "reason": "No face detected"
        }

    # 2. Embedding extraction
    embedding = face_extractor.get_embedding(aligned)
    if embedding is None:
        return {
            "verdict": False,
            "reason": "Embedding failed"
        }

    # 3. Identity matching
    users = db.fetch_all_users()
    if not users:
        return {
            "verdict": False,
            "reason": "No enrolled users"
        }

    best_score = -1.0
    best_user = None

    for user in users:
        result = matcher.verify(user["embedding"], embedding)
        if result["score"] > best_score:
            best_score = result["score"]
            best_user = user["name"]

    # 4. Temporal liveness
    is_live, live_score, details = liveness.check_liveness(frame)

    # Not enough temporal evidence yet
    if len(liveness.gray_buffer) < MIN_LIVE_FRAMES:
        return {
            "verdict": False,
            "reason": "Collecting liveness evidence",
            "identity_score": round(best_score, 4),
            "liveness_score": round(live_score, 4)
        }

    # Hard reject if spoof
    if not is_live:
        return {
            "verdict": False,
            "reason": "Liveness failed",
            "identity_score": round(best_score, 4),
            "liveness_score": round(live_score, 4),
            "liveness_details": details
        }

    # 5. Final fusion
    verdict = fuse(best_score, live_score)

    return {
        "verdict": verdict,
        "user": best_user if verdict else None,
        "identity_score": round(best_score, 4),
        "liveness_score": round(live_score, 4),
        "liveness_details": details
    }