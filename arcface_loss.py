import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss for face recognition.
    Enforces angular margin between classes.
    """

    def __init__(self, embed_dim, num_classes, margin=0.5, scale=64.0):
        super().__init__()
        self.margin = margin
        self.scale = scale

        # Class centers (one per identity)
        self.weight = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        """
        embeddings: (B, embed_dim), L2-normalized
        labels: (B,)
        """

        # Normalize class centers
        W = F.normalize(self.weight, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, W)  # (B, num_classes)
        cosine = cosine.clamp(-1.0, 1.0)

        # Add angular margin to target class
        theta = torch.acos(cosine)
        target_cosine = torch.cos(theta + self.margin)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = cosine * (1 - one_hot) + target_cosine * one_hot
        logits *= self.scale

        loss = F.cross_entropy(logits, labels)
        return loss
