import torch
import torch.optim as optim
from siamese_vit import ViTBackbone
from arcface_loss import ArcFaceLoss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Backbone (pretrained ViT)
    model = ViTBackbone(embed_dim=512).to(device)

    # ArcFace loss
    loss_fn = ArcFaceLoss(
        embed_dim=512,
        num_classes=10   # placeholder
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    # ---- Dummy batch (temporary) ----
    images = torch.randn(4, 3, 224, 224).to(device)
    labels = torch.tensor([0, 1, 2, 3]).to(device)
    # --------------------------------

    model.train()
    embeddings = model(images)
    loss = loss_fn(embeddings, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("✅ ArcFace + ViT training step works")
    print("Loss:", loss.item())


if __name__ == "__main__":
    main()
