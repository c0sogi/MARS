import os
import sys
import numpy as np
import pandas as pd
import torch

from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood
from library.data import get_dataloaders
from library.model import PGARNet
from library.train import run_training
from library.predict import generate_submission


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Initialize directories and device
    Config.setup()

    # Override Config for Fast Baseline Execution
    Config.EPOCHS = 6
    Config.MAX_TRAIN_SAMPLES = 1500  # Limit training data for speed
    Config.BATCH_SIZE = 16

    # Ensure reproducibility
    seed_everything(Config.SEED)

    print("=== Configuration ===")
    print(f"Device: {Config.DEVICE}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Max Train Samples: {Config.MAX_TRAIN_SAMPLES}")
    print("=====================")

    # ==========================================
    # 2. Training
    # ==========================================
    print("\n=== Starting Training ===")
    # run_training handles the loop, checkpointing, and returns best val score
    run_training(epochs=Config.EPOCHS, load_cached_data=True)

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("\n=== Validation & Failure Analysis ===")

    # Load the best model checkpoint
    model = PGARNet()
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading best model from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=Config.DEVICE)
    model.load_state_dict(state_dict)
    model.to(Config.DEVICE)
    model.eval()

    # Get Validation Data
    # Ensure we validate on the full validation set (disable subsampling for validation)
    Config.MAX_VAL_SAMPLES = None
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Containers for analysis
    all_targets = []
    all_preds = []
    all_sigmas = []

    # Metadata containers for correlation analysis
    meta_weeks = []
    meta_percent = []
    meta_age = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(val_loader):
            # Move inputs to device
            axial = inputs["axial"].to(Config.DEVICE)
            coronal = inputs["coronal"].to(Config.DEVICE)
            tabular = inputs["tabular"].to(Config.DEVICE)
            dt = inputs["dt"].to(Config.DEVICE)
            base_fvc = inputs["base_fvc"].to(Config.DEVICE)
            target = target.to(Config.DEVICE)

            # Forward pass
            fvc_pred, sigma_pred = model(axial, coronal, tabular, dt, base_fvc)

            # Store results (move to CPU numpy)
            all_targets.append(target.cpu().numpy())
            all_preds.append(fvc_pred.cpu().numpy())
            all_sigmas.append(sigma_pred.cpu().numpy())

            # Store metadata features
            # dt corresponds to 'Weeks' (relative time)
            meta_weeks.append(dt.cpu().numpy())
            # Tabular: [Age_norm, Percent_norm, ...]
            meta_age.append(tabular[:, 0].cpu().numpy())
            meta_percent.append(tabular[:, 1].cpu().numpy())

    # Concatenate all batches
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    sigma = np.concatenate(all_sigmas)

    weeks = np.concatenate(meta_weeks)
    age = np.concatenate(meta_age)
    percent = np.concatenate(meta_percent)

    # Compute Final Metric
    final_metric_tensor = laplace_log_likelihood(y_true, y_pred, sigma)
    final_metric = final_metric_tensor.item()

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation of Absolute Error with Features
    errors = np.abs(y_true - y_pred)

    print("\n--- Error Correlation Analysis ---")

    def calc_corr(name, feature_arr):
        if len(np.unique(feature_arr)) > 1:
            # np.corrcoef returns matrix [[1, r], [r, 1]]
            r = np.corrcoef(errors, feature_arr)[0, 1]
            print(f"Correlation (Error vs {name}): {r:.6f}")
        else:
            print(f"Correlation (Error vs {name}): N/A (Constant feature)")

    calc_corr("Weeks (Time)", weeks)
    calc_corr("Age (Normalized)", age)
    calc_corr("Percent (Normalized)", percent)
    calc_corr("Predicted Confidence (Sigma)", sigma)

    # ==========================================
    # 4. Submission
    # ==========================================
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
