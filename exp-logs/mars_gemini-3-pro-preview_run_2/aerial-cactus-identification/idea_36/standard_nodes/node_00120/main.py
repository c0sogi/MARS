import sys
import os
import numpy as np
import torch
from scipy.stats import pearsonr

# Import from the provided library
from library.config import Config
from library.utils import calculate_roc_auc, load_checkpoint, set_seed
from library.dataset import get_dataloaders
from library.model import UltraWideECARepNeXt
from library.train import run_training


def perform_failure_analysis(images, labels, preds):
    """
    Analyzes the correlation between prediction errors and image meta-features.

    Args:
        images (np.ndarray): Raw images (N, H, W, C) in uint8.
        labels (np.ndarray): Ground truth labels.
        preds (np.ndarray): Predicted probabilities.
    """
    # Calculate absolute error
    errors = np.abs(labels - preds)

    # Extract meta-features
    # Normalize to 0-1 for stats calculation
    imgs_float = images.astype(np.float32) / 255.0

    # 1. Brightness (Mean Intensity)
    brightness = np.mean(imgs_float, axis=(1, 2, 3))

    # 2. Contrast (Standard Deviation)
    contrast = np.std(imgs_float, axis=(1, 2, 3))

    # 3. Channel Means
    red_mean = np.mean(imgs_float[:, :, :, 0], axis=(1, 2))
    green_mean = np.mean(imgs_float[:, :, :, 1], axis=(1, 2))
    blue_mean = np.mean(imgs_float[:, :, :, 2], axis=(1, 2))

    features = {
        "Brightness": brightness,
        "Contrast": contrast,
        "Red Mean": red_mean,
        "Green Mean": green_mean,
        "Blue Mean": blue_mean,
    }

    print("\n--- Failure Analysis ---")
    print("Correlation between Error Magnitude and Image Features:")

    for name, feat_values in features.items():
        # Handle potential constant values to avoid NaNs
        if np.std(feat_values) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(feat_values, errors)
        print(f"{name}: {corr:.8f}")


def main():
    # Ensure reproducibility
    set_seed(42)

    # 1. Run Training Pipeline
    # This handles training 5 seeds, TTA inference on test set, and saving submission.csv
    print("Starting Training Pipeline...")
    run_training()

    # 2. Validation Assessment
    print("\nStarting Validation Assessment...")
    device = torch.device(Config.DEVICE)

    # Load Validation Data
    # Use cached data since training likely generated it
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Access raw data for analysis and labels for metric
    val_images = val_loader.dataset.images
    val_labels = val_loader.dataset.labels

    # Accumulate predictions from all seeds for a robust validation metric
    val_preds_accumulator = np.zeros(len(val_labels))

    for seed in Config.SEEDS:
        # Initialize model structure
        model = UltraWideECARepNeXt().to(device)

        # Load the trained checkpoint
        ckpt_path = Config.get_model_save_path(seed)
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint for seed {seed} not found. Skipping.")
            continue

        load_checkpoint(model, ckpt_path, device)

        # Switch to inference mode (fuse Conv+BN and branches)
        model.switch_to_deploy()
        model.eval()

        seed_preds = []

        # Inference Loop
        # We perform standard inference (no TTA) on validation for efficiency
        with torch.no_grad():
            for images_batch, _, _ in val_loader:
                images_batch = images_batch.to(device)
                logits = model(images_batch)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                seed_preds.extend(probs)

        val_preds_accumulator += np.array(seed_preds)

    # Average predictions across seeds
    final_val_preds = val_preds_accumulator / len(Config.SEEDS)

    # 3. Calculate and Print Final Metric
    # Printing full precision as requested
    final_auc = calculate_roc_auc(val_labels, final_val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 4. Failure Analysis
    perform_failure_analysis(val_images, val_labels, final_val_preds)


if __name__ == "__main__":
    main()
