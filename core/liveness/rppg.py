import numpy as np

def rppg_score(face_frames):
    # crude physiological proxy: green-channel stability
    greens = []

    for frame in face_frames:
        greens.append(np.mean(frame[:, :, 1]))

    variance = np.var(greens)

    # stable illumination → live
    score = np.exp(-variance / 5.0)
    return float(score)
