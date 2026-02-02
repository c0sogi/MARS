import os
import torch
import numpy as np
import scipy.stats as stats
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.trainer import run_curriculum_training, set_seed
from library.inference import generate_submission
from library.dataset import extract_patches, DenoisingDataset
from library.network import CAResDnCNN


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Fast Baseline
    # -------------------------------------------------------------------------
    # We modify the Config class attributes to create a fast execution profile
    # suitable for a baseline run within the time limit.

    # Limit training epochs
    Config.MAX_EPOCHS_STAGE_1 = 5
    Config.MAX_EPOCHS_STAGE_2 = 5

    # Increase stride to reduce the number of training patches (faster epoch)
    Config.STRIDE_STAGE_1 = 80
    Config.STRIDE_STAGE_2 = 60

    # Use distinct cache files for this fast run to avoid conflicts
    Config.CACHE_FILE_STAGE_1 = os.path.join(
        Config.WORKING_DIR, "train_patches_s1_fast.npy"
    )
    Config.CACHE_TARGETS_STAGE_1 = os.path.join(
        Config.WORKING_DIR, "train_targets_s1_fast.npy"
    )
    Config.CACHE_FILE_STAGE_2 = os.path.join(
        Config.WORKING_DIR, "train_patches_s2_fast.npy"
    )
    Config.CACHE_TARGETS_STAGE_2 = os.path.join(
        Config.WORKING_DIR, "train_targets_s2_fast.npy"
    )

    # Set seed for reproducibility
    set_seed(Config.SEED)

    print("Configuration set for fast baseline run.")
    print(
        f"Stage 1 Stride: {Config.STRIDE_STAGE_1}, Epochs: {Config.MAX_EPOCHS_STAGE_1}"
    )
    print(
        f"Stage 2 Stride: {Config.STRIDE_STAGE_2}, Epochs: {Config.MAX_EPOCHS_STAGE_2}"
    )

    # -------------------------------------------------------------------------
    # 2. Training Pipeline
    # -------------------------------------------------------------------------
    print("\n--- Starting Training ---")
    # This function handles data loading, model init, and the two-stage training loop
    run_curriculum_training()

    # -------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Starting Validation and Failure Analysis ---")
    device = torch.device(Config.DEVICE)

    # Load Validation Data
    # We use the standard validation stride from Config
    val_patches, val_targets = extract_patches(
        metadata_path=Config.VAL_METADATA_PATH,
        stride=Config.VAL_STRIDE,
        patch_size=Config.PATCH_SIZE,
        cache_patches_path=Config.CACHE_FILE_VAL,
        cache_targets_path=Config.CACHE_TARGETS_VAL,
        load_cached_data=True,
        is_test=False,
    )

    val_dataset = DenoisingDataset(val_patches, val_targets, augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load the Best Model
    model = CAResDnCNN(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        num_features=Config.NUM_FEATURES,
        num_blocks=Config.NUM_BLOCKS,
    ).to(device)

    if os.path.exists(Config.CHECKPOINT_PATH):
        print(f"Loading checkpoint from {Config.CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, map_location=device))
    else:
        print("Warning: Checkpoint not found. Using initialized weights.")

    model.eval()

    mse_sum = 0.0
    total_pixels = 0

    # Lists to store data for failure analysis
    all_inputs = []
    all_errors = []

    with torch.no_grad():
        for inputs, clean_targets in val_loader:
            inputs = inputs.to(device)
            clean_targets = clean_targets.to(device)

            # Forward pass (predict noise)
            noise_pred = model(inputs)

            # Reconstruct clean image
            clean_pred = inputs - noise_pred
            clean_pred = torch.clamp(clean_pred, 0.0, 1.0)

            # Calculate metrics
            diff = clean_pred - clean_targets
            batch_mse = torch.sum(diff**2)
            mse_sum += batch_mse.item()
            total_pixels += inputs.numel()

            # Collect data for analysis
            # Flatten and move to CPU
            batch_inputs_flat = inputs.cpu().numpy().flatten()
            batch_errors_flat = torch.abs(diff).cpu().numpy().flatten()

            all_inputs.append(batch_inputs_flat)
            all_errors.append(batch_errors_flat)

    # Calculate Final RMSE
    final_rmse = np.sqrt(mse_sum / total_pixels)
    print(f"Final Validation Metric: {final_rmse}")

    # Failure Analysis: Correlation
    flat_inputs = np.concatenate(all_inputs)
    flat_errors = np.concatenate(all_errors)

    corr, _ = stats.pearsonr(flat_inputs, flat_errors)
    print(f"Correlation (Input Intensity vs Error Magnitude): {corr}")

    # Interpret Correlation
    if corr > 0.1:
        print(
            "Analysis: Positive correlation indicates higher errors in brighter (background) regions."
        )
    elif corr < -0.1:
        print(
            "Analysis: Negative correlation indicates higher errors in darker (text) regions."
        )
    else:
        print(
            "Analysis: Errors are fairly uniformly distributed across intensity levels."
        )

    # -------------------------------------------------------------------------
    # 4. Conditional Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.011577641381826402

    if final_rmse < THRESHOLD:
        print(
            f"\nMetric {final_rmse} < Threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission()
    else:
        print(f"\nMetric {final_rmse} >= Threshold {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
