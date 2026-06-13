import yaml
import torch
import numpy as np
from pathlib import Path
import time

from core.identity import ViTFaceEmbedder
from core.liveness import (
    texture_score,
    temporal_score,
    rppg_score,
    fuse_scores
)
from core.identity.database import load_db, add_identity_with_embedding
from core.liveness.face_roi import extract_face_roi



class DwarapalVerifier:
    """
    Unified Identity + Liveness Verification Engine
    (Database-based identity, temporal liveness, PS-aligned)
    """

    def __init__(self, config_path, identity_ckpt):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Identity model (inference only)
        self.identity_model = ViTFaceEmbedder()
        ckpt_path = Path(identity_ckpt)
        if ckpt_path.exists():
            self.identity_model.load_state_dict(
                torch.load(identity_ckpt, map_location=self.device, weights_only=True)
            )
        else:
            print(f"[WARN] Identity checkpoint not found: {identity_ckpt}")
            print("[WARN] Identity verification will use untrained weights.")
        self.identity_model.to(self.device)
        self.identity_model.eval()

        self.tau_id = self.cfg["identity"]["threshold"]
        self.tau_live = self.cfg["liveness"]["threshold"]
        self.window_size = self.cfg["liveness"]["window_size"]

        # Temporal buffers
        self.buffer = []
        self.live_scores = []
        self.start_time = None

    # -------------------------
    # Reset temporal state
    # -------------------------
    def reset_liveness(self):
        self.buffer.clear()
        self.live_scores.clear()
        self.start_time = None

    # -------------------------
    # Frame accumulation
    # -------------------------
    def add_frame(self, frame_rgb):
        if self.start_time is None:
            self.start_time = time.time()

        face = extract_face_roi(frame_rgb)

        if face is not None:
            self.buffer.append(face)

            if len(self.buffer) > self.window_size:
                self.buffer.pop(0)



    # -------------------------
    # Identity (DATABASE-BASED)
    # -------------------------
    @torch.no_grad()
    def identify(self, live_face_tensor, face_embedding=None):
        embeddings, labels = load_db()

        if embeddings.shape[0] == 0:
            return "Unknown", 0.0

        if face_embedding is not None:
            z_live = np.array(face_embedding).reshape(1, -1)
        else:
            z_live = self.identity_model(
                live_face_tensor.to(self.device)
            ).cpu().numpy()  # (1, 512)

        sims = embeddings @ z_live.T
        idx = int(np.argmax(sims))
        score = float(sims[idx][0])

        if score >= self.tau_id:
            return labels[str(idx)], score

        return "Unknown", score

    # -------------------------
    # Liveness evaluation (TEMPORAL)
    # -------------------------
    def evaluate_liveness(self):
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

        # keep evidence window bounded
        if len(self.live_scores) > self.window_size:
            self.live_scores.pop(0)

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

    # -------------------------
    # New User Enrollment (PHASE 5.7)
    # -------------------------
    @torch.no_grad()
    def enroll_new_user(self, name, face_tensors):
        """
        face_tensors: list of (1,3,224,224) tensors
        """
        embeddings = []

        for x in face_tensors:
            z = self.identity_model(x.to(self.device))
            embeddings.append(z.cpu().numpy())

        mean_embedding = np.mean(
            np.vstack(embeddings),
            axis=0,
            keepdims=True
        )

        add_identity_with_embedding(name, mean_embedding)
