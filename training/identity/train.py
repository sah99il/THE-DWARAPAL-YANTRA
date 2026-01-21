import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from core.identity import ViTFaceEmbedder, ArcFaceHead
from data.loaders.identity_dataset import IdentityDataset
from data.augmentations.id_degradation import degrade_id_image

# --------------------
# Load Config
# --------------------
with open("training/identity/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------
# Dataset
# --------------------
transform = transforms.Compose([
    transforms.Resize((cfg["dataset"]["image_size"], cfg["dataset"]["image_size"])),
    transforms.ToTensor()
])

dataset = IdentityDataset(
    root_dir=cfg["dataset"]["root_dir"],
    transform=transform,
    degrade_fn=degrade_id_image
)

loader = DataLoader(
    dataset,
    batch_size=cfg["training"]["batch_size"],
    shuffle=True,
    num_workers=0   # REQUIRED on Windows
)

num_classes = len(dataset.identities)

# --------------------
# Model
# --------------------
embedder = ViTFaceEmbedder(
    model_name=cfg["model"]["backbone"],
    embed_dim=cfg["model"]["embedding_dim"]
).to(device)

arcface = ArcFaceHead(
    embedding_dim=cfg["model"]["embedding_dim"],
    num_classes=num_classes,
    scale=cfg["arcface"]["scale"],
    margin=cfg["arcface"]["margin"]
).to(device)

# --------------------
# Optimizer
# --------------------
optimizer = torch.optim.AdamW(
    list(embedder.parameters()) + list(arcface.parameters()),
    lr=cfg["training"]["learning_rate"],
    weight_decay=cfg["training"]["weight_decay"]
)

# --------------------
# Training Loop
# --------------------
os.makedirs(cfg["checkpoint"]["save_dir"], exist_ok=True)

for epoch in range(cfg["training"]["num_epochs"]):
    embedder.train()
    arcface.train()

    total_loss = 0.0

    for batch in loader:
        id_imgs = batch["id_image"].to(device)
        labels = batch["label"].to(device)

        embeddings = embedder(id_imgs)
        logits = arcface(embeddings, labels)

        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

    torch.save(
        embedder.state_dict(),
        os.path.join(cfg["checkpoint"]["save_dir"], f"embedder_epoch_{epoch+1}.pth")
    )
