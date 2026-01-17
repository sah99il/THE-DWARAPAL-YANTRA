import cv2
from services.enrollment_service import enroll_from_frame

cap = cv2.VideoCapture(0)

print("Capturing frame for enrollment...")
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to access camera")
    exit(1)

name = input("Enter your name for enrollment: ")
result = enroll_from_frame(frame, name)
print(result)