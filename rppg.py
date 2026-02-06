import numpy as np
from scipy.signal import butter, filtfilt


class RPPGDetector:
    def __init__(self, fps=30):
        self.fps = fps

        # Heart rate band: 0.7–4 Hz (42–240 BPM)
        self.low = 0.7
        self.high = 4.0

    def _bandpass(self, signal):
        nyq = 0.5 * self.fps
        low = self.low / nyq
        high = self.high / nyq

        b, a = butter(3, [low, high], btype="band")
        return filtfilt(b, a, signal)

    def extract_signal(self, frames):
        """
        frames: list of aligned face images (BGR)
        """
        green_vals = []

        for frame in frames:
            green = frame[:, :, 1]   # Green channel
            green_vals.append(np.mean(green))

        signal = np.array(green_vals)
        signal = signal - np.mean(signal)
        return signal

    def is_live(self, frames, std_thresh=0.02):
        """
        frames: ~3–5 seconds of face frames
        """
        if len(frames) < self.fps * 2:
            return False, 0.0

        raw_signal = self.extract_signal(frames)
        filtered = self._bandpass(raw_signal)

        signal_std = np.std(filtered)

        live = signal_std > std_thresh
        return live, signal_std

