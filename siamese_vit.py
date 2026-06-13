import torch
import torch.nn as nn
import timm


class ViTBackbone(nn.Module):
    """
    Vision Transformer backbone for face embedding.
    Uses pretrained weights.
    """

    def __init__(self, embed_dim=512):
        super().__init__()

        # Load pretrained ViT
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0  # removes classification head
        )

        vit_out_dim = self.vit.num_features

        # Projection head (face embedding)
        self.embedding = nn.Sequential(
            nn.Linear(vit_out_dim, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )

    def forward(self, x):
        features = self.vit(x)
        embeddings = self.embedding(features)
        embeddings = nn.functional.normalize(embeddings, dim=1)
        return embeddings


class SiameseViT(nn.Module):
    """
    Siamese network with shared ViT backbone.
    """

    def __init__(self, embed_dim=512):
        super().__init__()
        self.backbone = ViTBackbone(embed_dim)

    def forward(self, img1, img2):
        emb1 = self.backbone(img1)
        emb2 = self.backbone(img2)
        return emb1, emb2


# -------------------------
# Sanity check (no data needed)
# -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SiameseViT(embed_dim=512).to(device)
    model.eval()

    # Dummy aligned face inputs
    img1 = torch.randn(1, 3, 224, 224).to(device)
    img2 = torch.randn(1, 3, 224, 224).to(device)

    with torch.no_grad():
        e1, e2 = model(img1, img2)

    print("✅ Siamese ViT forward pass works")
    print("Embedding shape:", e1.shape)
