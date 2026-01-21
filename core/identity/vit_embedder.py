# core/identity/vit_embedder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ViTFaceEmbedder(nn.Module):
    """
    Vision Transformer based face embedding model.
    Outputs L2-normalized 512-D embeddings.
    """

    def __init__(self, model_name="vit_base_patch16_224", embed_dim=512):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0
        )

        self.embedding = nn.Linear(self.backbone.num_features, embed_dim)

    def forward(self, x):
        features = self.backbone(x)          # (B, D)
        embeddings = self.embedding(features)
        embeddings = F.normalize(embeddings, dim=1)
        return embeddings
