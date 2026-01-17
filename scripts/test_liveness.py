import cv2
from core.liveness.liveness_engine import LivenessDetector

cap = cv2.VideoCapture(0)
detector = LivenessDetector()

print("Look at camera for 3 seconds...")

for _ in range(60):
    ret, frame = cap.read()
    if not ret:
        break
    result = detector.check_liveness(frame)

cap.release()

print(result)