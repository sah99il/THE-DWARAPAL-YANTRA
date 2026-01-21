import torch
from core.identity import ViTFaceEmbedder, ArcFaceHead

model = ViTFaceEmbedder()
x = torch.randn(2, 3, 224, 224)

with torch.no_grad():
    emb = model(x)

print("Embedding shape:", emb.shape)
print("Embedding norms:", emb.norm(dim=1))

assert emb.shape == (2, 512)
assert torch.allclose(emb.norm(dim=1), torch.ones(2), atol=1e-4)

print("PHASE 2.1 CHECK PASSED ✅")
