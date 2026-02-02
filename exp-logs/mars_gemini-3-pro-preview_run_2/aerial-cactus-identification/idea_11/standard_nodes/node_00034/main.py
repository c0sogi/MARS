import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

from library.config import Config
from library.utils import set_seed, get_device, load_checkpoint
from library.model import NarrowSEResNet
from library.dataset import CactusDataset, get_transforms
from library.train import train_model
from library.inference import run_inference


def main():
    # ==========================================
    # 1. Model Training (Ensemble)
    # ==========================================
    print("Starting Ensemble Training...")
    for seed in Config.SEEDS:
        train_model(seed)
    print("Ensemble Training Complete.")

    # ==========================================
    # 2. Ensemble Validation
    # ==========================================
    print("\n--- Running Ensemble Validation ---")
    device = get_device()

    # Load Validation Data
    val_dataset = CactusDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        split="val",
        transform=get_transforms("val"),
        load_cached_data=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Collect Targets
    # We iterate the loader to ensure alignment with predictions
    all_targets = []
    for _, labels, _ in val_loader:
        all_targets.extend(labels.numpy())
    targets = np.array(all_targets)

    # Accumulate Predictions from all seeds
    num_samples = len(val_dataset)
    accumulated_probs = np.zeros((num_samples, 1), dtype=np.float32)

    for seed in Config.SEEDS:
        model_path = os.path.join(Config.WORK_DIR, f"model_seed_{seed}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model for seed {seed} missing. Skipping.")
            continue

        model = NarrowSEResNet().to(device)
        load_checkpoint(model_path, model, device=device)
        model.eval()

        seed_probs = []
        with torch.no_grad():
            for images, _, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                seed_probs.extend(probs.cpu().numpy())

        accumulated_probs += np.array(seed_probs)

    # Average probabilities
    avg_probs = accumulated_probs / len(Config.SEEDS)
    avg_probs = avg_probs.flatten()

    # Compute Metric
    val_auc = roc_auc_score(targets, avg_probs)
    print(f"Final Validation Metric: {val_auc:.15f}")

    # ==========================================
    # 3. Failure Analysis
    # ==========================================
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(targets - avg_probs)

    # Extract meta-features from raw images (uint8, 0-255)
    # val_dataset.images is [N, 32, 32, 3]
    raw_images = val_dataset.images

    # Vectorized calculation of meta-features
    meta_brightness = np.mean(raw_images, axis=(1, 2, 3))
    meta_contrast = np.std(raw_images, axis=(1, 2, 3))
    meta_red = np.mean(raw_images[..., 0], axis=(1, 2))
    meta_green = np.mean(raw_images[..., 1], axis=(1, 2))
    meta_blue = np.mean(raw_images[..., 2], axis=(1, 2))

    # Calculate correlations
    print("Correlation between Error Magnitude and Input Features:")

    def print_corr(name, feature):
        corr, _ = pearsonr(errors, feature)
        print(f"{name.ljust(12)}: {corr:.4f}")

    print_corr("Brightness", meta_brightness)
    print_corr("Contrast", meta_contrast)
    print_corr("Red Mean", meta_red)
    print_corr("Green Mean", meta_green)
    print_corr("Blue Mean", meta_blue)

    # ==========================================
    # 4. Submission
    # ==========================================
    # We use a threshold of 0.5 to determine validity, as AUC > 1.0 is impossible.
    if val_auc > 0.5:
        print("\nValidation metric satisfactory. Generating submission...")
        run_inference()
    else:
        print(f"\nValidation metric {val_auc} is too low. Submission skipped.")


if __name__ == "__main__":
    main()
