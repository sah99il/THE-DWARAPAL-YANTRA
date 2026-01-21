import numpy as np
import torch
from torch.utils.data import DataLoader
from core.identity import ViTFaceEmbedder
from data.loaders.identity_dataset import IdentityDataset
from data.augmentations.id_degradation import degrade_id_image
from torchvision import transforms

# Similarity Function
def cosine_similarity(a, b):
    return np.sum(a * b, axis=1)

# Score Generation
@torch.no_grad()
def compute_scores(model, dataloader, device):
    model.eval()

    genuine_scores = []
    impostor_scores = []

    for batch in dataloader:
        id_imgs = batch["id_image"].to(device)
        live_imgs = batch["live_image"].to(device)
        labels = batch["label"].cpu().numpy()

        z_id = model(id_imgs).cpu().numpy()
        z_live = model(live_imgs).cpu().numpy()

        for i in range(len(labels)):
            # genuine
            genuine_scores.append(
                cosine_similarity(z_id[i:i+1], z_live[i:i+1])[0]
            )

            # impostor (random other)
            j = np.random.choice([k for k in range(len(labels)) if labels[k] != labels[i]])
            impostor_scores.append(
                cosine_similarity(z_id[i:i+1], z_live[j:j+1])[0]
            )

    return np.array(genuine_scores), np.array(impostor_scores)


# FAR / TAR Computation
def compute_far_tar(genuine, impostor, threshold):
    far = np.mean(impostor >= threshold)
    tar = np.mean(genuine >= threshold)
    return far, tar


# Threshold Search
def find_threshold(genuine, impostor, target_far=1e-4):
    thresholds = np.sort(impostor)[::-1]

    for t in thresholds:
        far, tar = compute_far_tar(genuine, impostor, t)
        if far <= target_far:
            return t, far, tar

    return None, None, None
