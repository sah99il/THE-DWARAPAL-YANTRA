import numpy as np
import cv2
from insightface.app import FaceAnalysis

# Initialize InsightFace for face recognition only
# Detection is disabled because faces are already aligned
app = FaceAnalysis(
    name='buffalo_l',
    allowed_modules=['recognition']
)

# ctx_id = 0 for GPU, -1 for CPU
app.prepare(ctx_id=0)


def get_embedding(aligned_face: np.ndarray) -> np.ndarray:
    # Return None if no face is provided
    if aligned_face is None:
        return None

    # Ensure correct input size (112x112 RGB face)
    if aligned_face.shape != (112, 112, 3):
        raise ValueError("Face must be 112x112x3")

    # Convert BGR (OpenCV) to RGB (InsightFace)
    face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

    # Convert to float32 for model input
    face = face.astype(np.float32)

    # Change format from HWC to CHW
    face = np.transpose(face, (2, 0, 1))

    # Add batch dimension
    face = np.expand_dims(face, axis=0)

    # Extract face embedding
    embedding = app.models['recognition'].get_feat(face)[0]

    # Normalize embedding for cosine similarity
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


def compare_embeddings(e1: np.ndarray, e2: np.ndarray) -> float:
    # Return 0 if any embedding is missing
    if e1 is None or e2 is None:
        return 0.0

    # Cosine similarity for normalized vectors
    return float(np.dot(e1, e2))