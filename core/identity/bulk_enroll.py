import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import cv2
import numpy as np

from core.identity.database import add_identity, load_db
from core.identity.vit_embedder import ViTFaceEmbedder


DATA_DIR = "data/identity"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess(img, size=224):
    img = cv2.resize(img, (size, size))
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    return img / 255.0


def bulk_enroll():
    model = ViTFaceEmbedder().to(DEVICE)
    model.eval()

    embeddings, labels = load_db()
    existing_names = set(labels.values())

    for person in sorted(os.listdir(DATA_DIR)):
        person_dir = os.path.join(DATA_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        # Skip already enrolled identities
        if person in existing_names:
            continue

        imgs = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".png"))
        ]

        if len(imgs) == 0:
            continue

        # Use first image for enrollment
        img_path = os.path.join(person_dir, imgs[0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x = preprocess(img).to(DEVICE)

        with torch.no_grad():
            emb = model(x).cpu().numpy()

        add_identity(person, emb)
        print(f"[ENROLLED] {person}")

    print("Bulk enrollment complete.")


if __name__ == "__main__":
    bulk_enroll()
