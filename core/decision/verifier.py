import yaml
import torch
import numpy as np

from core.identity import ViTFaceEmbedder
from core.liveness import (
    texture_score,
    temporal_score,
    rppg_score,
    fuse_scores
)

class DwarapalVerifier:
    """
    Unified Identity + Liveness Verification Engine
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
        self.id_embedding = None

    # -------------------------
    # Identity (once per session)
    # -------------------------
    @torch.no_grad()
    def enroll_identity(self, face_tensor):
        emb = self.identity_model(face_tensor.to(self.device))
        self.id_embedding = emb.cpu()

        # Reset cached identity score
        if hasattr(self, "cached_identity_score"):
            del self.cached_identity_score

        return self.id_embedding


    # -------------------------
    # Frame accumulation
    # -------------------------
    def add_frame(self, frame_rgb):
        self.buffer.append(frame_rgb)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

    # -------------------------
    # Liveness evaluation
    # -------------------------
    def evaluate_liveness(self):
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

        return s_live

    # -------------------------
    # Final decision
    # -------------------------
    @torch.no_grad()
    def verify(self, live_face_tensor):
        """
        Runs identity check ONCE per session
        and liveness periodically.
        """
        assert self.id_embedding is not None, "Identity not enrolled"

        # -------------------------
        # Identity similarity (ONCE)
        # -------------------------
        if not hasattr(self, "cached_identity_score"):
            z_live = self.identity_model(live_face_tensor.to(self.device)).cpu()
            self.cached_identity_score = torch.cosine_similarity(
                self.id_embedding, z_live
            ).item()

        # -------------------------
        # Liveness (periodic)
        # -------------------------
        s_live = self.evaluate_liveness()

        accept = (
            self.cached_identity_score >= self.tau_id and
            s_live >= self.tau_live
        )

        return {
            "identity_score": self.cached_identity_score,
            "liveness_score": s_live,
            "decision": "ACCEPT" if accept else "REJECT"
        }

