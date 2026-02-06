import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="liveness_data",
        help="Dataset root containing train/ and val/ folders (ImageFolder format).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input-size", type=int, default=160)
    parser.add_argument("--out", default="enrolled_users/liveness_model.pt")
    args = parser.parse_args()

    # Expected layout:
    #   liveness_data/
    #     train/live/*.jpg
    #     train/spoof/*.jpg
    #     val/live/*.jpg
    #     val/spoof/*.jpg
    data_root = Path(args.data)
    train_root = data_root / "train"
    val_root = data_root / "val"
    if not train_root.exists() or not val_root.exists():
        raise FileNotFoundError(
            "Missing dataset folders. Expected:\n"
            "  liveness_data/train/live, liveness_data/train/spoof,\n"
            "  liveness_data/val/live,   liveness_data/val/spoof\n"
        )

    from torchvision import datasets, transforms

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(args.input_size + 32),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_ds = datasets.ImageFolder(str(train_root), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_root), transform=val_tf)

    # Enforce class naming for consistent labels:
    # ImageFolder uses alphabetical order -> live:0, spoof:1
    if train_ds.classes != ["live", "spoof"] or val_ds.classes != ["live", "spoof"]:
        raise ValueError(
            f"Expected class folders ['live', 'spoof'].\n"
            f"train classes: {train_ds.classes}\n"
            f"val classes:   {val_ds.classes}\n"
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)
    print("[INFO] Train samples:", len(train_ds), " Val samples:", len(val_ds))

    from liveness_net import SpoofNet

    model = SpoofNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            # y: 0=live, 1=spoof
            x = x.to(device)
            y = y.to(device).float().unsqueeze(1)

            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += float(loss.item()) * x.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += int((preds == y).sum().item())
            train_total += int(x.size(0))

        train_loss /= max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device).float().unsqueeze(1)
                logits = model(x)
                loss = loss_fn(logits, y)

                val_loss += float(loss.item()) * x.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += int((preds == y).sum().item())
                val_total += int(x.size(0))

        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        print(
            f"[EPOCH {epoch:03d}] "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model_state": model.state_dict(),
                "input_size": int(args.input_size),
                "mean": mean,
                "std": std,
            }
            torch.save(ckpt, out_path)
            print("[INFO] Saved:", out_path)

    print("[DONE] Best val loss:", best_val)


if __name__ == "__main__":
    main()

