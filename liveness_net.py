import torch
import torch.nn as nn


class SpoofNet(nn.Module):
    """
    Simple anti-spoof CNN.

    Output: single logit for "spoof" probability.
      - target 0: live
      - target 1: spoof
    """

    def __init__(self):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def load_spoof_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SpoofNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    meta = {
        "input_size": int(ckpt.get("input_size", 160)),
        "mean": tuple(float(x) for x in ckpt.get("mean", (0.5, 0.5, 0.5))),
        "std": tuple(float(x) for x in ckpt.get("std", (0.5, 0.5, 0.5))),
    }
    return model, meta

