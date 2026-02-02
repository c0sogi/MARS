import sys
import os
import torch
import numpy as np
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.train import run_training
from library.inference import generate_submission
from library.dataset import DenoisingDataset
from library.model import ResUNet
from library.utils import get_device, calculate_rmse, set_seed


def main():
    # --- 1. Configuration & Setup ---
    # Configure for high-performance run
    # Increasing epochs and patch density to maximize convergence
    Config.EPOCHS = 100
    Config.PATCHES_PER_IMAGE = 80

    # Ensure reproducibility
    set_seed(Config.SEED)
    device = get_device()

    print(f"Running on device: {device}")
    print(
        f"Configuration: Epochs={Config.EPOCHS}, Patches/Img={Config.PATCHES_PER_IMAGE}"
    )

    # --- 2. Training ---
    # Execute the training pipeline
    # The run_training function handles dataset creation, model initialization, and the training loop.
    run_training(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        load_cached_data=Config.LOAD_CACHED_DATA,
    )

    # --- 3. Validation & Failure Analysis ---
    print("Starting validation and failure analysis...")

    # Load the best model saved during training
    model = ResUNet().to(device)
    if os.path.exists(Config.MODEL_PATH):
        state_dict = torch.load(Config.MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    model.eval()

    # Prepare Validation Dataset
    val_dataset = DenoisingDataset(
        metadata_file=Config.VAL_METADATA,
        mode="val",
        load_cached_data=Config.LOAD_CACHED_DATA,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Containers for global metric calculation
    all_preds = []
    all_targets = []

    # Containers for failure analysis (flattened pixel data)
    flat_errors = []
    flat_noisy = []
    flat_clean = []

    with torch.no_grad():
        for noisy, residual in val_loader:
            noisy = noisy.to(device)
            residual = residual.to(device)

            # Inference
            pred_residual = model(noisy)

            # Store predictions and targets for global RMSE calculation
            all_preds.append(pred_residual.cpu())
            all_targets.append(residual.cpu())

            # --- Failure Analysis Data Collection ---
            # Calculate Clean Images for error analysis
            # Clean = Noisy - Residual
            pred_clean = torch.clamp(noisy - pred_residual, 0.0, 1.0)
            true_clean = torch.clamp(noisy - residual, 0.0, 1.0)

            # Absolute error per pixel
            abs_error = torch.abs(pred_clean - true_clean)

            # Flatten and store for correlation analysis
            flat_errors.append(abs_error.cpu().numpy().flatten())
            flat_noisy.append(noisy.cpu().numpy().flatten())
            flat_clean.append(true_clean.cpu().numpy().flatten())

    # Concatenate results from all batches
    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)

    # Calculate Final Metric (RMSE)
    val_rmse = calculate_rmse(y_true, y_pred)
    print(f"Final Validation Metric: {val_rmse}")

    # Perform Failure Analysis
    all_errors_np = np.concatenate(flat_errors)
    all_noisy_np = np.concatenate(flat_noisy)
    all_clean_np = np.concatenate(flat_clean)

    # Calculate correlations between error magnitude and pixel intensities
    corr_noisy, _ = pearsonr(all_errors_np, all_noisy_np)
    corr_clean, _ = pearsonr(all_errors_np, all_clean_np)

    print("Failure Analysis Results:")
    print(f"Correlation (Error vs Input Noisy Intensity): {corr_noisy}")
    print(f"Correlation (Error vs Target Clean Intensity): {corr_clean}")

    # --- 4. Submission Generation ---
    # Threshold defined in the task description
    THRESHOLD = 0.009138691164531186

    if val_rmse < THRESHOLD:
        print(
            f"Validation RMSE ({val_rmse}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"Validation RMSE ({val_rmse}) is above threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
