import numpy as np
from core.face.aligner import align_face
from core.face.extractor import FaceExtractor
from database.db_manager import DatabaseManager

face_extractor = FaceExtractor()
db = DatabaseManager()

def enroll_from_frame(frame, user_name: str):
    if not user_name:
        return {"success": False, "reason": "Empty name"}

    aligned = align_face(frame)
    if aligned is None:
        return {"success": False, "reason": "No face detected"}

    embedding = face_extractor.get_embedding(aligned)
    if embedding is None:
        return {"success": False, "reason": "Embedding failed"}

    # Store as bytes
    db.add_user(user_name, embedding.astype(np.float32))

    return {
        "success": True,
        "user": user_name
    }