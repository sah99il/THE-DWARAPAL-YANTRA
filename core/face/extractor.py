import numpy as np
import cv2
from insightface.app import FaceAnalysis

class FaceExtractor:
    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            root="./models",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        if aligned_face is None:
            return None

        # 🔑 IMPORTANT: DO NOT re-detect face
        # Instead, pass aligned face directly to recognition model
        blob = cv2.dnn.blobFromImage(
            aligned_face,
            scalefactor=1.0 / 127.5,
            size=(112, 112),
            mean=(127.5, 127.5, 127.5),
            swapRB=True
        )

        # Run only recognition model
        embedding = self.app.models["recognition"].session.run(
            None,
            {self.app.models["recognition"].input_name: blob}
        )[0][0]

        embedding = embedding / np.linalg.norm(embedding)
        return embedding