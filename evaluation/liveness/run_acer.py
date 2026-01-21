from torch.utils.data import DataLoader
from data.loaders.liveness_dataset import LivenessDataset
from evaluation.liveness.acer import evaluate_liveness, find_liveness_threshold

dataset = LivenessDataset(
    root_dir="data/liveness",
    window_size=150
)

scores, labels = evaluate_liveness(dataset)

tau_live, acer = find_liveness_threshold(scores, labels)

print(f"Liveness Threshold τ_live: {tau_live:.3f}")
print(f"ACER: {acer:.4f}")
