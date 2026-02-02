import sys
import os
import torch
import numpy as np
import pandas as pd

# Import from the provided library
from library.config import Config
import library.train as train_lib
import library.inference as inference_lib
from library.utils import set_seed, load_checkpoint
from library.dataset import get_dataloaders
from library.model import UNet


def main():
    # --- 1. Configuration for Fast Baseline ---
    # We override the default configuration to ensure the task completes within the time limit.
    # Reducing the ensemble to a single model and limiting epochs allows for a fast run
    # while still aiming for convergence on the small dataset.
    print("Configuring for fast baseline run...")

    # Use a single seed for the baseline to save time (vs 10 models)
    Config.ENSEMBLE_SEEDS = [42]

    # Set epochs to 500. On an A100 with ~92 images, this is very fast (minutes)
    # and ensures sufficient convergence to potentially meet the strict RMSE threshold.
    Config.NUM_EPOCHS = 500
    Config.SCHEDULER_T_MAX = 500

    # Ensure reproducibility
    set_seed(42)

    # Initialize directories
    Config.initialize()

    # --- 2. Training ---
    print("Starting training pipeline...")
    train_lib.train_ensemble()

    # --- 3. Validation & Failure Analysis ---
    print("Starting validation and failure analysis...")

    # Get validation dataloader (utilizing cache if available)
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Initialize model and load the trained checkpoint
    device = torch.device(Config.DEVICE)
    model = UNet(n_channels=Config.IN_CHANNELS, n_classes=Config.OUT_CHANNELS)
    model.to(device)

    # Load the specific seed we trained
    seed = Config.ENSEMBLE_SEEDS[0]
    model_path = Config.get_model_path(seed)

    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    load_checkpoint(model_path, model, device=device)
    model.eval()

    # metrics accumulation
    val_sse_accum = 0.0  # Sum of Squared Errors
    total_pixels = 0

    # Failure analysis data
    img_errors = []
    img_means = []
    img_stds = []
    img_areas = []

    # Inference loop (No Grad)
    with torch.no_grad():
        for noisy_imgs, clean_imgs, _ in val_loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            # Forward pass
            outputs = model(noisy_imgs)

            # Calculate pixel-wise difference
            diff = outputs - clean_imgs

            # Update global metrics
            val_sse_accum += torch.sum(diff**2).item()
            total_pixels += clean_imgs.numel()

            # Per-image analysis
            # Calculate RMSE for this specific image
            img_mse = torch.mean(diff**2).item()
            img_rmse = np.sqrt(img_mse)
            img_errors.append(img_rmse)

            # Extract features from input (noisy image)
            # Move to CPU numpy for stats
            input_np = noisy_imgs.cpu().numpy().squeeze()

            img_means.append(np.mean(input_np))
            img_stds.append(np.std(input_np))
            img_areas.append(input_np.size)

    # Compute Final Global RMSE
    if total_pixels > 0:
        final_rmse = np.sqrt(val_sse_accum / total_pixels)
    else:
        final_rmse = float("inf")

    # --- 4. Reporting ---
    # Print the exact metric format required
    print(f"Final Validation Metric: {final_rmse}")

    # Failure Analysis: Correlation
    print("-" * 30)
    print("Failure Analysis Results")
    print("-" * 30)

    if len(img_errors) > 1:
        corr_mean = np.corrcoef(img_errors, img_means)[0, 1]
        corr_std = np.corrcoef(img_errors, img_stds)[0, 1]
        corr_area = np.corrcoef(img_errors, img_areas)[0, 1]

        print(f"Correlation (Error vs Input Mean): {corr_mean:.4f}")
        print(f"Correlation (Error vs Input Std): {corr_std:.4f}")
        print(f"Correlation (Error vs Image Area): {corr_area:.4f}")
    else:
        print("Insufficient validation samples for correlation analysis.")

    # --- 5. Submission ---
    THRESHOLD = 0.011870221132053216

    if final_rmse < THRESHOLD:
        print(
            f"\nMetric {final_rmse} < {THRESHOLD}. Proceeding to submission generation..."
        )
        # inference_lib.predict_and_save uses the Config.ENSEMBLE_SEEDS we set earlier
        inference_lib.predict_and_save()
    else:
        print(f"\nMetric {final_rmse} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
