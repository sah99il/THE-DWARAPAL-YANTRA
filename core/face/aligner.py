import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Singleton model (VERY IMPORTANT for performance)
_face_app = None

def _get_face_app():
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(name="buffalo_l", root="./models")
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


def align_face(frame):
    """
    Detects the largest face using InsightFace and returns a 112x112 aligned crop.
    Returns None if no face detected.
    """
    app = _get_face_app()
    faces = app.get(frame)

    if len(faces) == 0:
        return None

    # Choose largest face (important for multi-face scenes)
    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    x1, y1, x2, y2 = map(int, face.bbox)
    h, w, _ = frame.shape

    # Safety clamp
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    aligned = cv2.resize(crop, (112, 112))
    return aligned