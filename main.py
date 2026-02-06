import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = get_device()

    print("🏛️ Sahil ka Dwarpal")
    print("🔹 Torch version:", torch.__version__)
    print("🔹 Device:", device)

    if device.type == "cuda":
        print("🔹 GPU:", torch.cuda.get_device_name(0))

    print("✅ System ready for biometric pipeline")


if __name__ == "__main__":
    main()


