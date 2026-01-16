import numpy as np
import cv2
from insightface.app import FaceAnalysis

# Load ONLY the recognition model
app = FaceAnalysis(
    name='buffalo_l',
    allowed_modules=['recognition']
)
app.prepare(ctx_id=0)  # -1 for CPU


def get_embedding(aligned_face: np.ndarray) -> np.ndarray:
    """
    aligned_face: 112x112 BGR image (OpenCV)
    returns: normalized 512-D embedding
    """

    if aligned_face is None:
        return None

    if aligned_face.shape != (112, 112, 3):
        raise ValueError("Face must be 112x112x3")

    # BGR -> RGB
    face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

    # Convert to float32
    face = face.astype(np.float32)

    # HWC -> CHW
    face = np.transpose(face, (2, 0, 1))

    # Add batch dimension
    face = np.expand_dims(face, axis=0)

    # Get embedding
    embedding = app.models['recognition'].get_feat(face)[0]

    # Normalize (MANDATORY)
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


def compare_embeddings(e1: np.ndarray, e2: np.ndarray) -> float:
    """
    Cosine similarity between two normalized embeddings
    """
    if e1 is None or e2 is None:
        return 0.0
    return float(np.dot(e1, e2))