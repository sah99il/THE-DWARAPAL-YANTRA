import numpy as np
import cv2
import onnxruntime as ort
import os

# Define the path to the ONNX recognition model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'buffalo_l', 'w600k_r50.onnx')

class FaceExtractor:
    def __init__(self):
        # Initialize ONNX runtime session
        self.session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def get_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        # Return None if no face is provided
        if aligned_face is None:
            return None

        # Ensure correct input size (112x112 RGB face)
        if aligned_face.shape != (112, 112, 3):
            raise ValueError("Face must be 112x112x3")

        # Convert BGR (OpenCV) to RGB
        face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

        # Convert to float32 for model input
        face = face.astype(np.float32)

        # Change format from HWC to CHW
        face = np.transpose(face, (2, 0, 1))

        # Add batch dimension
        face = np.expand_dims(face, axis=0)

        # Perform inference
        embedding = self.session.run([self.output_name], {self.input_name: face})[0]

        # Normalize embedding for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)

        return embedding[0] # Return the first (and only) embedding

def compare_embeddings(e1: np.ndarray, e2: np.ndarray) -> float:
    # Return 0 if any embedding is missing
    if e1 is None or e2 is None:
        return 0.0

    # Cosine similarity for normalized vectors
    return float(np.dot(e1, e2))