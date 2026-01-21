import cv2

class Camera:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self):
        self.cap.release()
