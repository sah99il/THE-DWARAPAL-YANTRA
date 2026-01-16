import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

detector = MTCNN()

def align_face(image):
    """
    Detects and aligns the most confident face to 112x112 using 5-point similarity transform.
    """

    if image is None or image.size == 0:
        return None

    # Convert to RGB for MTCNN
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb)

    if not results:
        return None

    best_face = max(results, key=lambda r: r['confidence'])

    if best_face['confidence'] < 0.95:
        return None

    keypoints = best_face['keypoints']

    detected_points = np.array([
        keypoints['left_eye'],
        keypoints['right_eye'],
        keypoints['nose'],
        keypoints['mouth_left'],
        keypoints['mouth_right']
    ], dtype=np.float32)

    ref_points = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]
    ], dtype=np.float32)

    # ArcFace offset correction
    ref_points[:, 0] += 8.0

    m, _ = cv2.estimateAffinePartial2D(
        detected_points.reshape(-1, 1, 2),
        ref_points.reshape(-1, 1, 2)
    )

    if m is None:
        return None

    aligned_face = cv2.warpAffine(
        image,
        m,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return aligned_face