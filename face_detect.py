import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    providers=['CPUExecutionProvider']
)

class FaceDetector:
    """
    High-performance face detector using SCRFD (InsightFace).
    Outputs bounding box + 5 facial landmarks.
    """

    def __init__(self, device="cuda"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.detector = FaceAnalysis(
            name="buffalo_l",   # includes SCRFD detector
            providers=providers
        )

        ctx_id = 0 if device == "cuda" else -1
        self.detector.prepare(ctx_id=ctx_id)

    def detect(self, image):
        """
        Detect faces in a BGR image.

        Returns:
            list of faces with attributes:
            - bbox (4,)
            - kps (5,2)
        """
        return self.detector.get(image)


def visualize(image, faces):
    """
    Draw bounding boxes and landmarks (debug / demo only).
    """
    vis = image.copy()

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for (x, y) in face.kps.astype(int):
            cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)

    return vis


# -------------------------
# CLI Test
# -------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python face_detect.py input.jpg output.jpg")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        raise FileNotFoundError("Input image not found")

    detector = FaceDetector(device="cuda")
    faces = detector.detect(img)

    if len(faces) == 0:
        print("❌ No face detected")
        sys.exit(0)

    vis = visualize(img, faces)
    cv2.imwrite(sys.argv[2], vis)

    print(f"✅ Faces detected: {len(faces)}")
    print("🔹 Landmarks of first face:\n", faces[0].kps)
