import cv2
import numpy as np
from skimage.feature import local_binary_pattern


# -------------------------
# LBP Entropy
# -------------------------
def lbp_entropy(gray):
    lbp = local_binary_pattern(
        gray,
        P=8,
        R=1,
        method="uniform"
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 11),
        density=True
    )

    hist += 1e-6  # numerical stability
    entropy = -np.sum(hist * np.log(hist))
    return entropy


# -------------------------
# FFT High-Frequency Ratio
# -------------------------
def fft_high_freq_ratio(gray):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    h, w = gray.shape
    cx, cy = h // 2, w // 2

    low_freq_radius = min(h, w) // 8

    mask = np.ones((h, w), np.uint8)
    cv2.circle(mask, (cy, cx), low_freq_radius, 0, -1)

    high_freq_energy = np.sum(magnitude * mask)
    total_energy = np.sum(magnitude) + 1e-6

    return high_freq_energy / total_energy


# -------------------------
# FINAL TEXTURE SCORE
# -------------------------
def texture_score(face_rgb):
    """
    Input: face ROI (RGB)
    Output: texture confidence ∈ [0, 1]
    """
    if face_rgb is None or face_rgb.size == 0:
        return 0.0

    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)

    # Reject extremely flat regions
    if gray.std() < 5:
        return 0.0

    entropy = lbp_entropy(gray)
    hf_ratio = fft_high_freq_ratio(gray)

    # --- Proper normalization ---
    # Live skin entropy ≈ [1.5 – 3.5]
    entropy_score = np.clip((entropy - 1.0) / 2.5, 0, 1)

    # Screens have excessive high-frequency energy
    freq_score = np.clip(1.0 - (hf_ratio - 0.15) / 0.35, 0, 1)

    score = 0.6 * entropy_score + 0.4 * freq_score
    return float(score)
