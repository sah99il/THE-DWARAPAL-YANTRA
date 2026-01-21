import cv2
import os
import random
import torch
from torch.utils.data import Dataset

class LivenessDataset(Dataset):
    def __init__(self, root_dir, window_size=150):
        self.samples = []
        self.window = window_size

        for label, cls in enumerate(["spoof", "live"]):
            cls_dir = os.path.join(root_dir, cls)
            for vid in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, vid), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        cap = cv2.VideoCapture(path)

        frames = []
        while len(frames) < self.window:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (112, 112))
            frame = frame[:, :, ::-1]  # BGR → RGB
            frames.append(frame)

        cap.release()

        frames = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0

        return {
            "frames": frames,
            "label": label
        }
