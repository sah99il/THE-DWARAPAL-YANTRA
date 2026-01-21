# core/identity/arcface_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceHead(nn.Module):
    """
    ArcFace head with angular margin.
    Used ONLY during training.
    """

    def __init__(self, embedding_dim, num_classes, scale=64.0, margin=0.5):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.W)

        self.s = scale
        self.m = margin

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        W = F.normalize(self.W, dim=1)

        cos_theta = F.linear(embeddings, W)
        theta = torch.acos(torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7))

        target_logits = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = (one_hot * target_logits) + ((1.0 - one_hot) * cos_theta)
        output *= self.s

        return output
