import cv2
import numpy as np

def temporal_score(face_frames):
    diffs = []

    for i in range(1, len(face_frames)):
        f1 = face_frames[i-1].astype(float)
        f2 = face_frames[i].astype(float)
        diffs.append(np.mean(np.abs(f2 - f1)))

    mean_motion = np.mean(diffs)

    # normalize motion consistency
    score = np.exp(-mean_motion / 20.0)
    return float(score)
