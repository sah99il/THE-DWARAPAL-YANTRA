import cv2
import numpy as np
from scipy import signal
import mediapipe as mp

class LivenessDetector:
    def __init__(self, buffer_size=150, fps=30):
        # Buffer for pulse signal (~5 seconds)
        self.buffer_size = buffer_size
        self.fps = fps

        self.green_means = []
        self.prev_gray = None
        self.blink_count = 0

        # MediaPipe Face Mesh for landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # Eye landmark indices (MediaPipe)
        self.left_eye_idx = [33, 160, 158, 133, 153, 144]
        self.right_eye_idx = [362, 385, 387, 263, 373, 380]

    def extract_forehead_roi(self, frame, landmarks):
        # Extract small ROI around forehead center
        h, w, _ = frame.shape
        forehead = landmarks[151]
        x, y = int(forehead.x * w), int(forehead.y * h)

        return frame[max(0, y-20):min(h, y+20),
                     max(0, x-20):min(w, x+20)]

    def eye_aspect_ratio(self, eye_points):
        # Basic EAR calculation for blink detection
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        return (v1 + v2) / (2.0 * h + 1e-6)

    def check_liveness(self, frame):
        score = 0.0

        # Run face landmark detection
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return False, 0.0, 0.0

        landmarks = results.multi_face_landmarks[0].landmark

        # ---------- Pulse (rPPG) ----------
        roi = self.extract_forehead_roi(frame, landmarks)
        green_mean = np.mean(roi[:, :, 1])
        self.green_means.append(green_mean)

        # Maintain rolling buffer
        if len(self.green_means) > self.buffer_size:
            self.green_means.pop(0)
        else:
            return False, 0.0, 0.0

        # Normalize signal
        data = np.array(self.green_means)
        data = (data - np.mean(data)) / (np.std(data) + 1e-6)

        # Bandpass filter for human heart rate
        b, a = signal.butter(4, [0.7, 3.0], btype='bandpass', fs=self.fps)
        filtered = signal.filtfilt(b, a, data)

        # FFT to find dominant frequency
        fft_vals = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(filtered), 1 / self.fps)

        heart_rate = freqs[np.argmax(fft_vals)] * 60
        if 45 <= heart_rate <= 180:
            score += 0.5

        # ---------- Blink detection ----------
        h, w, _ = frame.shape
        left_eye = np.array(
            [(landmarks[i].x * w, landmarks[i].y * h) for i in self.left_eye_idx]
        )
        right_eye = np.array(
            [(landmarks[i].x * w, landmarks[i].y * h) for i in self.right_eye_idx]
        )

        ear = (self.eye_aspect_ratio(left_eye) +
               self.eye_aspect_ratio(right_eye)) / 2

        if ear < 0.20:
            self.blink_count += 1

        if self.blink_count >= 1:
            score += 0.25

        # ---------- Head motion ----------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is not None:
            motion = np.mean(cv2.absdiff(gray, self.prev_gray))
            if motion > 1.5:
                score += 0.25

        self.prev_gray = gray

        # Final decision
        is_live = score >= 0.6
        confidence = round(score, 2)

        return is_live, confidence, round(heart_rate, 1)
