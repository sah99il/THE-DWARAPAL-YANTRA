import cv2
import numpy as np

from core.face.aligner import align_face
from core.face.extractor import FaceExtractor
from database.db_manager import DatabaseManager


face_extractor = FaceExtractor()
db = DatabaseManager()


def enroll_from_id_image(image: np.ndarray, user_name: str):
    """
    Enrolls a user using a static ID card image.
    """

    if image is None:
        return {
            "success": False,
            "reason": "Invalid image"
        }

    if not user_name or len(user_name.strip()) == 0:
        return {
            "success": False,
            "reason": "Invalid user name"
        }

    # 1. Align face from ID image
    aligned = align_face(image)

    if aligned is None:
        return {
            "success": False,
            "reason": "No face detected on ID card"
        }

    # 2. Extract embedding
    embedding = face_extractor.get_embedding(aligned)

    if embedding is None:
        return {
            "success": False,
            "reason": "Failed to extract embedding"
        }

    # 3. Store in database
    db.add_user(user_name, embedding)

    return {
        "success": True,
        "user": user_name
    }