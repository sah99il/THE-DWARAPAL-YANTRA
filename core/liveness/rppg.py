import numpy as np
from scipy.signal import butter, filtfilt

def bandpass(signal, low=0.7, high=4.0, fs=30):
    b, a = butter(3, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, signal)

def rppg_score(frames_rgb):
    green_signal = []

    for frame in frames_rgb:
        green = frame[:, :, 1]
        green_signal.append(np.mean(green))

    green_signal = np.array(green_signal)

    if len(green_signal) < 30:
        return 0.0

    filtered = bandpass(green_signal)

    spectrum = np.abs(np.fft.rfft(filtered))
    peak = np.max(spectrum)
    noise = np.mean(spectrum) + 1e-6

    ratio = peak / noise
    score = np.clip(ratio / 5.0, 0, 1)

    return float(score)
