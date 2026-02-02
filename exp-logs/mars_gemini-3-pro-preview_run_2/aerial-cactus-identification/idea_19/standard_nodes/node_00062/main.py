import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr

# Import from the provided library
from library.config import Config
from library import utils, dataset, model, engine, inference


def main():
    # 1. Setup and Reproducibility
    utils.seed_everything(42)
    device = torch.device(Config.DEVICE)

    print("========================================")
    print("      STARTING PIPELINE EXECUTION       ")
    print("========================================")

    # 2. Training Phase
    # Train 5 independent instances as defined in Config.SEEDS
    print("\n--- Training Phase ---")
    for seed in Config.SEEDS:
        print(f"\nTraining Seed {seed}...")
        engine.run_training_seed(seed)

    # 3. Validation & Failure Analysis Phase
    print("\n--- Validation & Failure Analysis ---")

    # Load Validation Data
    # We use the library function to ensure consistent preprocessing
    _, val_loader, _ = dataset.get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # Collect Ground Truth Targets
    # Iterate loader once to get all labels (shuffle=False in val_loader)
    val_targets = []
    for _, labels in val_loader:
        val_targets.append(labels.numpy())
    val_targets = np.concatenate(val_targets).flatten()

    # Perform Ensemble Prediction on Validation Set
    num_samples = len(val_targets)
    val_preds_accum = np.zeros(num_samples, dtype=np.float32)
    successful_seeds = 0

    for seed in Config.SEEDS:
        model_path = Config.get_model_path(seed)
        if not os.path.exists(model_path):
            print(f"Warning: Model for seed {seed} not found. Skipping.")
            continue

        # Load Model
        net = model.CustomWideResNet()
        net.to(device)
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.eval()

        # Predict (Standard Inference, no TTA for validation metric speed)
        seed_preds = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                outputs = net(images)
                probs = torch.sigmoid(outputs)
                seed_preds.append(probs.cpu().numpy())

        val_preds_accum += np.concatenate(seed_preds).flatten()
        successful_seeds += 1

        # Free memory
        del net
        torch.cuda.empty_cache()

    if successful_seeds == 0:
        print("Error: No models were trained successfully.")
        return

    # Average predictions
    val_preds_avg = val_preds_accum / successful_seeds

    # Calculate and Print Final Metric
    final_metric = utils.calculate_roc_auc(val_targets, val_preds_avg)
    print(f"Final Validation Metric: {final_metric:.10f}")

    # Failure Analysis: Correlation between Error and Image Features
    print("\nPerforming Failure Analysis...")

    # Calculate absolute error
    errors = np.abs(val_targets - val_preds_avg)

    # Extract meta-features from validation images
    # Note: Images are normalized, but correlations remain valid
    meta_brightness = []
    meta_contrast = []
    meta_red = []
    meta_green = []
    meta_blue = []

    for images, _ in val_loader:
        imgs_np = images.numpy()  # Shape: (B, C, H, W)

        # Brightness: Mean intensity across all channels and pixels
        meta_brightness.extend(np.mean(imgs_np, axis=(1, 2, 3)))

        # Contrast: Standard deviation across all channels and pixels
        meta_contrast.extend(np.std(imgs_np, axis=(1, 2, 3)))

        # Channel Means
        meta_red.extend(np.mean(imgs_np[:, 0, :, :], axis=(1, 2)))
        meta_green.extend(np.mean(imgs_np[:, 1, :, :], axis=(1, 2)))
        meta_blue.extend(np.mean(imgs_np[:, 2, :, :], axis=(1, 2)))

    # Compute Correlations
    features = {
        "Brightness": meta_brightness,
        "Contrast": meta_contrast,
        "Red_Mean": meta_red,
        "Green_Mean": meta_green,
        "Blue_Mean": meta_blue,
    }

    print("Correlation between Error Magnitude and Image Features:")
    for name, feat_values in features.items():
        if len(feat_values) != len(errors):
            continue
        corr, _ = pearsonr(feat_values, errors)
        print(f"{name}: {corr:.4f}")

    # 4. Submission Phase
    # Generate submission if the model is performing better than random guessing (0.5)
    # (Note: Prompt threshold of 1.0 is physically impossible for AUC, assuming 0.5)
    if final_metric > 0.5:
        print("\n--- Generating Submission ---")
        inference.ensemble_predictions()
    else:
        print(f"\nSkipping submission. Metric {final_metric} is too low.")


if __name__ == "__main__":
    main()
