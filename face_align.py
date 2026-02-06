import cv2
import numpy as np

# Standard ArcFace reference (5 landmarks)
ARC_REF = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def _select_5_landmarks(landmarks):
    """
    Converts 68 / 106 landmarks → 5 landmarks
    """
    if landmarks.shape[0] == 5:
        return landmarks

    # 68 or 106 landmarks → select canonical points
    left_eye = landmarks[36]
    right_eye = landmarks[45]
    nose = landmarks[30]
    left_mouth = landmarks[48]
    right_mouth = landmarks[54]

    return np.array([
        left_eye,
        right_eye,
        nose,
        left_mouth,
        right_mouth
    ], dtype=np.float32)

def align_face(image, landmarks, output_size=(224, 224)):
    landmarks = np.asarray(landmarks, dtype=np.float32)

    if landmarks.ndim != 2 or landmarks.shape[1] != 2:
        raise ValueError("Invalid landmark format")

    landmarks = _select_5_landmarks(landmarks)

    ref = ARC_REF.copy()
    ref[:, 0] *= output_size[0] / 112
    ref[:, 1] *= output_size[1] / 112

    M, _ = cv2.estimateAffinePartial2D(landmarks, ref, method=cv2.LMEDS)
    if M is None:
        return None
    aligned = cv2.warpAffine(image, M, output_size)

    return aligned
