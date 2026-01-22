import torch
import numpy as np
from core.decision.verifier import DwarapalVerifier

verifier = DwarapalVerifier(
    config_path="configs/system.yaml",
    identity_ckpt="models/checkpoints/identity/embedder_epoch_10.pth"
)

fake_face = torch.randn(1, 3, 224, 224)
verifier.enroll_identity(fake_face)

for _ in range(150):
    fake_frame = (np.random.rand(224, 224, 3) * 255).astype("uint8")
    verifier.add_frame(fake_frame)

out = verifier.verify(fake_face)

print(out)
