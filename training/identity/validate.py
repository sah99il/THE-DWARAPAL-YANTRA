import torch
import numpy as np
from torch.utils.data import DataLoader
from core.identity import ViTFaceEmbedder

def cosine_similarity(a, b):
    return torch.sum(a * b, dim=1)

@torch.no_grad()
def validate(embedder, dataloader, device):
    embedder.eval()
    sims = []
    labels = []

    for batch in dataloader:
        z1 = embedder(batch["id_image"].to(device))
        z2 = embedder(batch["live_image"].to(device))

        sim = cosine_similarity(z1, z2)
        sims.extend(sim.cpu().numpy())
        labels.extend(batch["label"].cpu().numpy())

    return np.array(sims), np.array(labels)
