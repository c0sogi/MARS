import sys
import os
import warnings
import numpy as np
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, calculate_rmse
from library.model import UNet
from library.dataset import load_processed_data, DenoisingDataset
from library.train import run_training
from library.inference import generate_submission

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    set_seed()
    device = torch.device(Config.DEVICE)

    # 2. Training
    # We limit epochs to 30 to ensure the baseline runs quickly while still learning.
    # Batch size is kept at Config default (128) which is efficient for 64x64 patches.
    run_training(load_cached_data=True, epochs=30, batch_size=Config.BATCH_SIZE)

    # 3. Validation & Failure Analysis
    # Load Validation Data
    val_data = load_processed_data(
        Config.VAL_METADATA_PATH, "val", load_cached_data=True
    )
    val_dataset = DenoisingDataset(val_data, mode="val")
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    # Load Best Model
    model = UNet().to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print(f"Error: Model file not found at {Config.MODEL_SAVE_PATH}")
        return

    model.eval()

    all_preds_flat = []
    all_targets_flat = []

    # Lists for Failure Analysis
    img_errors = []
    img_means = []
    img_stds = []

    with torch.no_grad():
        for noisy, clean, _ in val_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Inference
            outputs = model(noisy)

            # Move to CPU for analysis
            pred_np = outputs.cpu().numpy()
            clean_np = clean.cpu().numpy()
            noisy_np = noisy.cpu().numpy()

            # Flatten and store for global RMSE
            all_preds_flat.append(pred_np.flatten())
            all_targets_flat.append(clean_np.flatten())

            # --- Failure Analysis Metrics ---
            # Calculate Mean Absolute Error for this specific image
            mae = np.mean(np.abs(pred_np - clean_np))
            img_errors.append(mae)

            # Calculate Input Features
            img_means.append(np.mean(noisy_np))
            img_stds.append(np.std(noisy_np))

    # Concatenate all pixels to compute Global RMSE
    total_preds = np.concatenate(all_preds_flat)
    total_targets = np.concatenate(all_targets_flat)

    # Calculate and Print Final Metric
    final_rmse = calculate_rmse(total_preds, total_targets)
    print(f"Final Validation Metric: {final_rmse}")

    # Compute Correlations for Failure Analysis
    if len(img_errors) > 1:
        corr_mean, _ = pearsonr(img_errors, img_means)
        corr_std, _ = pearsonr(img_errors, img_stds)

        print(f"Failure Analysis - Correlation (Error vs Input Mean): {corr_mean}")
        print(f"Failure Analysis - Correlation (Error vs Input Std): {corr_std}")
    else:
        print("Insufficient validation data for failure analysis.")

    # 4. Submission
    if final_rmse < 0.20155566930770874:
        generate_submission(load_cached_data=True)
    else:
        print(f"Validation RMSE ({final_rmse}) not low enough to generate submission.")


if __name__ == "__main__":
    main()
