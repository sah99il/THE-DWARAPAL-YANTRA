import numpy as np
from core.liveness import (
    texture_score,
    temporal_score,
    rppg_score,
    fuse_scores
)

def evaluate_liveness(dataset):
    """
    dataset yields:
    {
        'frames': Tensor[T,3,H,W],
        'label': int (1=live, 0=spoof)
    }
    """

    scores = []
    labels = []

    for sample in dataset:
        frames = sample["frames"].numpy().transpose(0, 2, 3, 1) * 255
        frames = frames.astype("uint8")

        s_texture = texture_score(frames[-1])
        s_temporal = temporal_score(frames)
        s_rppg = rppg_score(frames)

        s_live = fuse_scores(s_texture, s_temporal, s_rppg)

        scores.append(s_live)
        labels.append(sample["label"])

    return np.array(scores), np.array(labels)

def compute_apcer_bpcer(scores, labels, threshold):
    scores = np.array(scores)
    labels = np.array(labels)

    spoof = labels == 0
    live = labels == 1

    apcer = np.mean(scores[spoof] >= threshold)   # spoofs accepted
    bpcer = np.mean(scores[live] < threshold)     # lives rejected

    return apcer, bpcer

def find_liveness_threshold(scores, labels):
    thresholds = np.linspace(0, 1, 1000)

    best_acer = 1.0
    best_t = 0.5

    for t in thresholds:
        apcer, bpcer = compute_apcer_bpcer(scores, labels, t)
        acer = 0.5 * (apcer + bpcer)

        if acer < best_acer:
            best_acer = acer
            best_t = t

    return best_t, best_acer
