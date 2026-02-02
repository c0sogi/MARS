import os
import sys
import torch
import numpy as np
from scipy.stats import pearsonr

# Import provided library modules
import library.config as config
import library.train as train_lib
import library.dataset as dataset_lib
import library.utils as utils_lib
import library.inference as inference_lib
from library.model import CAResDnCNN


def main():
    # Ensure reproducibility
    utils_lib.seed_everything(config.SEED)

    # ---------------------------------------------------------
    # 1. Configuration Overrides for Fast Baseline
    # ---------------------------------------------------------
    # Modify training parameters to ensure execution within 2 hours.
    # Reducing epochs and increasing stride for dense sampling reduces computational load.
    train_lib.NUM_EPOCHS_STAGE_1 = 10
    train_lib.NUM_EPOCHS_STAGE_2 = 10
    dataset_lib.STRIDE_DENSE = (
        10  # Increase stride to reduce patch count (Speed up Stage 2)
    )

    # ---------------------------------------------------------
    # 2. Train Model
    # ---------------------------------------------------------
    # We use all available images (limit=None) but with reduced epochs/density.
    print("Starting training pipeline...")
    model = train_lib.train_model(limit=None)

    # ---------------------------------------------------------
    # 3. Validation & Metric Calculation
    # ---------------------------------------------------------
    print("Starting validation...")
    # Load validation data (Sparse stride is standard for validation)
    val_patches, val_targets = dataset_lib.get_processed_data(
        mode="val", stride_type="sparse", load_cached_data=True
    )

    # Prepare data for inference
    device = config.DEVICE
    val_patches_t = torch.from_numpy(val_patches).to(device)
    val_targets_t = torch.from_numpy(val_targets).to(device)

    # Run Inference
    model.eval()
    with torch.no_grad():
        # Predict noise
        noise_pred = model(val_patches_t)

        # Reconstruct clean image: Input - Noise
        clean_pred = val_patches_t - noise_pred

        # Clip to valid range [0, 1]
        clean_pred = torch.clamp(clean_pred, 0.0, 1.0)

    # Calculate RMSE
    # Move to CPU for numpy calculation
    clean_pred_np = clean_pred.cpu().numpy()
    val_targets_np = val_targets  # This is already numpy

    rmse = utils_lib.calculate_rmse(val_targets_np, clean_pred_np)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {rmse}")

    # ---------------------------------------------------------
    # 4. Failure Analysis
    # ---------------------------------------------------------
    print("Performing failure analysis...")
    # Calculate Error Magnitude: |Predicted - True|
    error_map = np.abs(clean_pred_np - val_targets_np)

    # Flatten arrays for correlation calculation
    error_flat = error_map.flatten()
    input_flat = val_patches.flatten()

    # Calculate Pearson Correlation
    corr, _ = pearsonr(error_flat, input_flat)
    print(
        f"Correlation between model's error magnitude and input pixel intensity: {corr}"
    )

    # ---------------------------------------------------------
    # 5. Submission Generation
    # ---------------------------------------------------------
    THRESHOLD = 0.011577641381826402

    if rmse < THRESHOLD:
        print(
            f"Validation metric {rmse} meets threshold {THRESHOLD}. Generating submission..."
        )
        inference_lib.generate_submission()
    else:
        print(
            f"Validation metric {rmse} does not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
