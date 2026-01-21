import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from core.identity import ViTFaceEmbedder
from data.loaders.identity_dataset import IdentityDataset
from data.augmentations.id_degradation import degrade_id_image
from evaluation.identity.far_tar import compute_scores, find_threshold

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = IdentityDataset(
    root_dir="data/identity",
    transform=transform,
    degrade_fn=degrade_id_image
)

loader = DataLoader(dataset, batch_size=32, shuffle=False)

model = ViTFaceEmbedder()
model.load_state_dict(torch.load("models/checkpoints/identity/embedder_epoch_10.pth"))
model.to(device)

genuine, impostor = compute_scores(model, loader, device)

tau, far, tar = find_threshold(genuine, impostor)

print(f"Selected Threshold τ: {tau:.4f}")
print(f"FAR: {far:.6f}")
print(f"TAR: {tar:.4f}")
