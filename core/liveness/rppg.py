import cv2
import numpy as np

def rppg_score(face_frames):
    # crude physiological proxy: green-channel stability
    greens = []

    for frame in face_frames:
        resized = cv2.resize(frame, (128, 128))
        greens.append(np.mean(resized[:, :, 1]))

    variance = np.var(greens)

    # stable illumination → live
    score = np.exp(-variance / 5.0)
    return float(score)
