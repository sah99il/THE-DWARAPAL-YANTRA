import cv2
import numpy as np

def temporal_score(frames_rgb):
    flows = []

    for i in range(len(frames_rgb) - 1):
        prev = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2GRAY)
        curr = cv2.cvtColor(frames_rgb[i + 1], cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale=0.5, levels=2,
            winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2,
            flags=0
        )

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flows.append(np.mean(mag))

    flows = np.array(flows)

    motion_var = np.var(flows)
    motion_mean = np.mean(flows)

    score = np.clip((motion_var + motion_mean) * 10.0, 0, 1)
    return float(score)
