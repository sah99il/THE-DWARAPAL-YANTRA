import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

detector = MTCNN()

def align_face(image):
    """
    Detects faces in an image, aligns them using a 5-point similarity transformation, and crops them.

    Args:
        image: The input image as a NumPy array.

    Returns:
        The aligned and cropped face as a 112x112 NumPy array, or None if no face is detected.
    """
    results = detector.detect_faces(image)
    if not results:
        return None

    # Use the face with the highest confidence
    best_face = max(results, key=lambda r: r['confidence'])
    
    keypoints = best_face['keypoints']
    
    # Standard 112x112 reference points
    ref_points = np.array([
        [30.2946, 51.6963],  # Left eye
        [65.5318, 51.5014],  # Right eye
        [48.0252, 71.7366],  # Nose
        [33.5493, 92.3655],  # Left mouth
        [62.7299, 92.2041]   # Right mouth
    ], dtype=np.float32)

    detected_points = np.array([
        keypoints['left_eye'],
        keypoints['right_eye'],
        keypoints['nose'],
        keypoints['mouth_left'],
        keypoints['mouth_right']
    ], dtype=np.float32)

    # Compute the transformation matrix
    # Note: cv2.estimateAffinePartial2D is for rigid transform (translation, rotation, scale)
    # For full affine, you might use cv2.estimateAffine2D
    # For similarity transform (which is what we want), estimateAffinePartial2D is a good choice.
    
    # We need to reshape the arrays for the function
    src_pts = detected_points.reshape(1, -1, 2)
    dst_pts = ref_points.reshape(1, -1, 2)

    # Use estimateAffinePartial2D for similarity transform
    m, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    
    if m is None:
        # Could not compute the transformation
        return None

    # Apply the affine transformation
    aligned_face = cv2.warpAffine(image, m, (112, 112), borderMode=cv2.BORDER_REPLICATE)

    return aligned_face