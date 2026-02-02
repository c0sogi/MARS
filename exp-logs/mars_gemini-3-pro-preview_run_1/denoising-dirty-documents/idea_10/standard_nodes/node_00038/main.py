import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.nn import MSELoss

# Import library modules
import library.config
import library.train
import library.inference
import library.dataset
import library.utils


def main():
    # =========================================================================
    # 1. Configuration Override for Fast Baseline
    # =========================================================================
    # We override defaults to ensure the run completes quickly (<2h) but trains enough to perform well.
    # 30 epochs * 2 models * 2 streams = ~120 epochs total serial equivalent.
    # With 92 images, this is extremely fast and fits well within the time limit.
    print("Configuring parameters for fast baseline run...")
    library.config.EPOCHS = 30
    library.config.ENSEMBLE_SIZE = 2
    library.config.BATCH_SIZE = 16

    # Ensure reproducibility
    library.utils.set_seed(42)

    # =========================================================================
    # 2. Training
    # =========================================================================
    print("Starting Training Phase...")
    # debug=False ensures we use our overridden config values (30 epochs)
    # instead of the hardcoded debug values (2 epochs).
    library.train.train_model(debug=False)

    # =========================================================================
    # 3. Validation & Failure Analysis
    # =========================================================================
    print("Starting Validation Phase...")
    device = library.config.DEVICE

    # Load the models we just trained
    models = library.inference.load_ensemble(device)

    if not models:
        print("Error: No models loaded. Aborting.")
        return

    # Get validation data
    # mode='train' returns (train_loader, val_loader)
    _, val_loader = library.dataset.get_dataloaders(
        batch_size=1, mode="train", load_cached_data=True
    )

    total_sse = 0.0  # Sum of Squared Errors
    total_pixels = 0

    # storage for failure analysis
    img_rmses = []
    img_means = []
    img_stds = []

    print(f"Validating on {len(val_loader)} images...")

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # Move data to device
            inputs = batch["input"].to(device)  # Shape: (1, 1, H, W)
            targets = batch["target"].to(device)  # Shape: (1, 1, H, W)

            # Predict using TTA (consistent with inference/submission)
            preds = library.inference.predict_with_tta(models, inputs)

            # Compute Error
            # Note: Inputs/Targets/Preds are all inverted (0=bg, 1=text).
            # RMSE is invariant to this inversion |(1-y) - (1-y_hat)| = |y_hat - y|
            diff = preds - targets
            squared_diff = diff**2

            # Update global stats
            batch_sse = torch.sum(squared_diff).item()
            n_pixels = inputs.numel()

            total_sse += batch_sse
            total_pixels += n_pixels

            # Update per-image stats for failure analysis
            # RMSE for this specific image
            img_mse = batch_sse / n_pixels
            img_rmse = np.sqrt(img_mse)
            img_rmses.append(img_rmse)

            # Image features (use CPU numpy)
            input_np = inputs.cpu().numpy()
            img_means.append(np.mean(input_np))
            img_stds.append(np.std(input_np))

    # Compute Final Metric
    final_mse = total_sse / total_pixels
    final_rmse = np.sqrt(final_mse)

    # Print exactly as required
    print(f"Final Validation Metric: {final_rmse}")

    # Failure Analysis
    print("-" * 30)
    print("Failure Analysis")
    print("-" * 30)

    if len(img_rmses) > 1:
        corr_mean = np.corrcoef(img_rmses, img_means)[0, 1]
        corr_std = np.corrcoef(img_rmses, img_stds)[0, 1]
        print(f"Correlation (RMSE vs Input Mean Intensity): {corr_mean}")
        print(f"Correlation (RMSE vs Input Std Dev): {corr_std}")
    else:
        print("Not enough validation samples for correlation analysis.")
    print("-" * 30)

    # =========================================================================
    # 4. Submission
    # =========================================================================
    THRESHOLD = 0.011870221132053216

    if final_rmse < THRESHOLD:
        print(
            f"Metric {final_rmse} is below threshold {THRESHOLD}. Generating submission..."
        )
        library.inference.generate_submission(debug=False)
    else:
        print(
            f"Metric {final_rmse} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
