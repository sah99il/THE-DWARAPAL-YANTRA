import yaml
import torch
import numpy as np
import time

from core.identity import ViTFaceEmbedder
from core.liveness import (
    texture_score,
    temporal_score,
    rppg_score,
    fuse_scores
)
from core.identity.database import load_db


class DwarapalVerifier:
    """
    Unified Identity + Liveness Verification Engine
    (Database-based identity, session-free)
    """

    def __init__(self, config_path, identity_ckpt):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Identity model (inference only)
        self.identity_model = ViTFaceEmbedder()
        self.identity_model.load_state_dict(
            torch.load(identity_ckpt, map_location=self.device)
        )
        self.identity_model.to(self.device)
        self.identity_model.eval()

        self.tau_id = self.cfg["identity"]["threshold"]
        self.tau_live = self.cfg["liveness"]["threshold"]
        self.window_size = self.cfg["liveness"]["window_size"]

        self.buffer = []

        self.live_scores = []
        self.start_time = None

    # -------------------------
    # Frame accumulation
    # -------------------------
    def add_frame(self, frame_rgb):
        if self.start_time is None:
            self.start_time = time.time()

        self.buffer.append(frame_rgb)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)


    # -------------------------
    # Identity (DATABASE-BASED)
    # -------------------------
    @torch.no_grad()
    def identify(self, live_face_tensor):
        """
        Identify person from identity database.
        Returns: (name, similarity_score)
        """
        embeddings, labels = load_db()

        if embeddings.shape[0] == 0:
            return "Unknown", 0.0

        z_live = self.identity_model(
            live_face_tensor.to(self.device)
        ).cpu().numpy()  # (1, 512)

        # cosine similarity (embeddings are normalized)
        sims = embeddings @ z_live.T  # (N, 1)

        idx = int(np.argmax(sims))
        score = float(sims[idx][0])

        if score >= self.tau_id:
            return labels[str(idx)], score

        return "Unknown", score

    # -------------------------
    # Liveness evaluation
    # -------------------------
    def evaluate_liveness(self):
        # Not enough frames yet
        if len(self.buffer) < self.window_size:
            return None, "COLLECTING_FRAMES"

        elapsed = time.time() - self.start_time
        if elapsed < self.cfg["liveness"]["min_seconds"]:
            return None, "WAITING_TIME"

        frames = np.array(self.buffer)

        s_texture = texture_score(frames[-1])
        s_temporal = temporal_score(frames)
        s_rppg = rppg_score(frames)

        s_live = fuse_scores(
            s_texture,
            s_temporal,
            s_rppg,
            self.cfg["fusion"]["w_texture"],
            self.cfg["fusion"]["w_temporal"],
            self.cfg["fusion"]["w_rppg"],
        )

        self.live_scores.append(s_live)

        # Stability check
        mean_score = float(np.mean(self.live_scores))
        variance = float(np.var(self.live_scores))

        if variance > self.cfg["liveness"]["max_variance"]:
            return mean_score, "UNSTABLE_SIGNAL"

        if mean_score >= self.tau_live:
            return mean_score, "LIVE_CONFIRMED"

        return mean_score, "SPOOF_SUSPECTED"

    # -------------------------
    # Final decision
    # -------------------------
    def verify(self, identity_score, liveness_result):
        liveness_score, liveness_state = liveness_result

        if liveness_state != "LIVE_CONFIRMED":
            return {
                "identity_score": identity_score,
                "liveness_score": liveness_score or 0.0,
                "decision": liveness_state
            }

        accept = identity_score >= self.tau_id

        return {
            "identity_score": identity_score,
            "liveness_score": liveness_score,
            "decision": "ACCEPT" if accept else "REJECT"
        }

