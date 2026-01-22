import numpy as np
import cv2
from skimage.feature import local_binary_pattern

# -----------------------------
# LBP PARAMETERS
# -----------------------------
LBP_POINTS = 8
LBP_RADIUS = 1

def lbp_entropy(gray):
    lbp = local_binary_pattern(
        gray, LBP_POINTS, LBP_RADIUS, method="uniform"
    )
    hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59), density=True)
    hist = hist + 1e-6
    entropy = -np.sum(hist * np.log(hist))
    return entropy

def fft_high_freq_ratio(gray):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    center = (h // 2, w // 2)

    mask = np.ones((h, w))
    cv2.circle(mask, center, min(h, w) // 6, 0, -1)

    high_freq_energy = np.sum(magnitude * mask)
    total_energy = np.sum(magnitude)

    return high_freq_energy / (total_energy + 1e-6)

def texture_score(face_rgb):
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)

    entropy = lbp_entropy(gray)
    hf_ratio = fft_high_freq_ratio(gray)

    # Normalize empirically
    entropy_score = np.clip(entropy / 5.0, 0, 1)
    freq_score = 1.0 - np.clip(hf_ratio * 5.0, 0, 1)

    score = 0.5 * entropy_score + 0.5 * freq_score
    return float(np.clip(score, 0, 1))
