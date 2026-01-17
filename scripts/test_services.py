import cv2
from services.verification_service import verify_frame

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
cap.release()

result = verify_frame(frame)
print(result)
