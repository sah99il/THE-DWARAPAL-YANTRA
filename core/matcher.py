import numpy as np

class FaceMatcher:
    def __init__(self, threshold=0.65):
        # Similarity threshold for ArcFace embeddings
        self.threshold = threshold

    def verify(self, embedding_id, embedding_live):
        # Compare ID image embedding with live selfie embedding

        if embedding_id is None or embedding_live is None:
            return {
                "match": False,
                "score": 0.0,
                "status": "Missing Embedding Data"
            }

        # Cosine similarity between two embeddings
        dot_product = np.dot(embedding_id, embedding_live)
        norm_id = np.linalg.norm(embedding_id)
        norm_live = np.linalg.norm(embedding_live)

        score = dot_product / (norm_id * norm_live)

        # Decide match based on threshold
        is_match = score >= self.threshold

        # Simple confidence labeling
        if score > 0.85:
            confidence = "High"
        elif score > 0.65:
            confidence = "Medium"
        else:
            confidence = "Low/No Match"

        return {
            "match": bool(is_match),
            "score": round(float(score), 4),
            "confidence": confidence,
            "status": "Success"
        }


def get_matcher_logic():
    # Create matcher with default project threshold
    return FaceMatcher(threshold=0.65)