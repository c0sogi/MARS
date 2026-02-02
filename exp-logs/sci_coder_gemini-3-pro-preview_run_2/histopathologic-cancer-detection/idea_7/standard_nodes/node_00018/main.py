import sys
import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed
from library.train import run_training
from library.inference import run_inference, load_ensemble
from library.dataset import load_data, PathologyDataset, get_transforms


def main():
    # 1. Setup and Configuration Overrides for Fast Baseline
    Config.setup()
    set_seed(Config.SEED)

    # Override Config for speed constraints (2 hours max)
    # We train 2 folds for each of the 2 architectures (4 models total)
    # 3 Epochs is sufficient for a strong baseline on this dataset size with pre-trained models
    Config.EPOCHS = 3
    Config.N_FOLDS = 2

    # 2. Run Training
    # This will train ConvNeXt-Tiny and EfficientNetV2-S for Folds 0 and 1
    print("--- Starting Training Phase ---")
    run_training()

    # 3. Validation and Failure Analysis
    print("\n--- Starting Validation & Failure Analysis ---")
    device = torch.device(Config.DEVICE)

    # Load the official hold-out validation set
    print("Loading validation data...")
    val_images, val_labels = load_data("val", load_cached_data=True)

    # Create DataLoader for validation
    val_dataset = PathologyDataset(
        val_images, val_labels, transforms=get_transforms("val"), is_test=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load the ensemble we just trained
    models = load_ensemble(device)

    if not models:
        print("Error: No models were trained or loaded. Exiting.")
        return

    # Run Inference on Validation Set
    print(f"Running inference on {len(val_dataset)} validation samples...")
    all_preds = []
    all_targets = []

    # We perform inference without TTA for validation to save time,
    # relying on the ensemble robustness.
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)

            # Ensemble prediction accumulator
            batch_accum = torch.zeros((inputs.size(0), 1), device=device)

            for model in models:
                outputs = model(inputs)
                probs = torch.sigmoid(outputs)
                batch_accum += probs

            # Average across models
            batch_avg = batch_accum / len(models)

            all_preds.extend(batch_avg.cpu().numpy().flatten())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate and Print Metric
    val_auc = roc_auc_score(all_targets, all_preds)
    print(f"Final Validation Metric: {val_auc}")

    # --- Failure Analysis ---
    print("\n--- Performing Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(all_targets - all_preds)

    # Extract image features efficiently using numpy
    # Normalize to 0-1
    imgs_norm = val_images.astype(np.float32) / 255.0

    # Calculate stats
    # Shape: (N, H, W, C) -> Mean over (H, W)
    means_rgb = imgs_norm.mean(axis=(1, 2))  # (N, 3)

    features = {
        "brightness": means_rgb.mean(axis=1),
        "red_mean": means_rgb[:, 0],
        "green_mean": means_rgb[:, 1],
        "blue_mean": means_rgb[:, 2],
        "contrast": imgs_norm.std(axis=(1, 2, 3)),  # Global std per image
    }

    print("Correlation between Error Magnitude and Input Features:")
    for name, values in features.items():
        corr, p_val = pearsonr(errors, values)
        print(f"  {name}: Correlation = {corr:.4f} (p-value = {p_val:.4f})")

    # 4. Submission
    # Threshold defined in task description
    THRESHOLD = 0.9889066475479729

    if val_auc > THRESHOLD:
        print(f"\nValidation Metric ({val_auc}) > Threshold ({THRESHOLD}).")
        print("Proceeding to generate submission...")

        # Configure TTA for Test Inference to maximize score
        # Using 4 views is a good trade-off for speed/accuracy within the remaining time
        Config.TTA_VIEWS = 4

        run_inference(load_cached_data=True)
    else:
        print(f"\nValidation Metric ({val_auc}) <= Threshold ({THRESHOLD}).")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
