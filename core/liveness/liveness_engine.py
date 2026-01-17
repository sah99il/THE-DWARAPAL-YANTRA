import cv2
import numpy as np
import mediapipe as mp
from collections import deque

class LivenessDetector:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_buffer = deque(maxlen=window_size)
        self.gray_buffer = deque(maxlen=window_size)

        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )

    def _motion_score(self):
        if len(self.gray_buffer) < 2:
            return 0.0

        diffs = []
        for i in range(1, len(self.gray_buffer)):
            diff = cv2.absdiff(self.gray_buffer[i], self.gray_buffer[i-1])
            diffs.append(np.mean(diff))

        return float(np.mean(diffs))

    def _texture_score(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())

    def _blink_detected(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return False

        # Eye landmarks (left eye)
        eye_ids = [33, 160, 158, 133, 153, 144]
        h, w, _ = frame.shape
        points = []

        for lm in results.multi_face_landmarks[0].landmark:
            points.append((int(lm.x * w), int(lm.y * h)))

        eye = [points[i] for i in eye_ids]

        # Compute Eye Aspect Ratio (EAR)
        def dist(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        ear = (dist(eye[1], eye[5]) + dist(eye[2], eye[4])) / (2.0 * dist(eye[0], eye[3]))
        return ear < 0.2  # blink threshold

    def check_liveness(self, frame):
        if frame is None:
            return False, 0.0, "No frame"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.gray_buffer.append(gray)
        self.frame_buffer.append(frame)

        motion = self._motion_score()
        texture = self._texture_score(frame)
        blink = self._blink_detected(frame)

        # Normalized scoring
        motion_score = min(motion / 5.0, 1.0)
        texture_score = min(texture / 100.0, 1.0)
        blink_score = 1.0 if blink else 0.0

        final_score = 0.4 * motion_score + 0.4 * texture_score + 0.2 * blink_score

        is_live = final_score > 0.5

        return is_live, final_score, {
            "motion": motion_score,
            "texture": texture_score,
            "blink": blink_score
        }