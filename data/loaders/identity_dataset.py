import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class IdentityDataset(Dataset):
    def __init__(self, root_dir, transform=None, degrade_fn=None):
        self.root_dir = root_dir
        self.transform = transform
        self.degrade_fn = degrade_fn

        self.identities = sorted(os.listdir(root_dir))
        self.samples = []

        for idx, pid in enumerate(self.identities):
            img_dir = os.path.join(root_dir, pid)
            imgs = os.listdir(img_dir)
            for img in imgs:
                self.samples.append((idx, os.path.join(img_dir, img)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, img_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        live_img = img
        id_img = self.degrade_fn(img) if self.degrade_fn else img

        if self.transform:
            live_img = self.transform(live_img)
            id_img = self.transform(id_img)

        return {
            "id_image": id_img,
            "live_image": live_img,
            "label": label
        }
