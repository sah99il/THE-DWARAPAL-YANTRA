import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.fft import fft2, fftshift


class LivenessDetector:
    def __init__(self):
        # LBP parameters
        self.radius = 1
        self.n_points = 8 * self.radius

    def lbp_score(self, gray_face):
        lbp = local_binary_pattern(
            gray_face,
            self.n_points,
            self.radius,
            method="uniform"
        )
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, self.n_points + 3),
            density=True
        )
        return np.var(hist)

    def fft_score(self, gray_face):
        f = fft2(gray_face)
        fshift = fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)
        return np.mean(magnitude)

    def is_live(self, face_bgr, lbp_thresh=0.02, fft_thresh=3.5):
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))

        lbp_val = self.lbp_score(gray)
        fft_val = self.fft_score(gray)

        live = (lbp_val > lbp_thresh) and (fft_val > fft_thresh)

        return live, {
            "lbp": lbp_val,
            "fft": fft_val
        }

