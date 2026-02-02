import sys
import os
import torch
import numpy as np
import pandas as pd

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import DP_GI_BiLSTM
from library.train import Trainer
from library.inference import predict


def run():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline Execution
    # 200 epochs is too long for a 2-hour limit.
    # 15 epochs with the full dataset (approx 2-3 mins per epoch on A100)
    # fits comfortably within the time limit while allowing convergence.
    Config.EPOCHS = 15
    print(f"Configuration updated: Running for {Config.EPOCHS} epochs.")

    # ==========================================
    # 2. Training
    # ==========================================
    print("\n=== Starting Training Phase ===")
    trainer = Trainer()
    trainer.fit()

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("\n=== Starting Validation & Failure Analysis ===")
    device = torch.device(Config.DEVICE)

    # Load the best model saved during training
    model = DP_GI_BiLSTM(input_dim=Config.INPUT_DIM).to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"Loaded best model from {Config.MODEL_PATH}")
    else:
        print("Warning: Best model not found. Using current model state.")

    model.eval()

    # Get Validation Loader
    # load_cached_data=True ensures we use the preprocessed data
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Containers for analysis
    all_preds = []
    all_targets = []
    all_u_out = []
    all_inputs = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for inputs, targets, u_out in val_loader:
            inputs = inputs.to(device)

            # Forward pass
            preds = model(inputs)

            # Move to CPU and flatten for analysis
            # Inputs shape: (B, L, F) -> (B*L, F)
            # Preds/Targets shape: (B, L) -> (B*L)
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(targets.cpu().numpy().flatten())
            all_u_out.append(u_out.cpu().numpy().flatten())
            all_inputs.append(inputs.cpu().numpy().reshape(-1, Config.INPUT_DIM))

    # Concatenate all batches
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    u_out_flat = np.concatenate(all_u_out)
    X_flat = np.concatenate(all_inputs, axis=0)

    # Filter for Inspiratory Phase (u_out == 0)
    insp_mask = u_out_flat == 0

    # Compute Metric
    if np.sum(insp_mask) > 0:
        # MAE on inspiratory phase
        diff = np.abs(y_pred[insp_mask] - y_true[insp_mask])
        final_metric = np.mean(diff)
    else:
        final_metric = 0.0

    # Print Required Metric String
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation of Error Magnitude with Features
    print("\n--- Failure Analysis (Inspiratory Phase) ---")
    print("Correlation between Absolute Error and Input Features:")

    errors_insp = np.abs(y_pred[insp_mask] - y_true[insp_mask])
    X_insp = X_flat[insp_mask]
    feature_names = Config.FEATURE_COLS

    for i, feat_name in enumerate(feature_names):
        feat_values = X_insp[:, i]

        # Check for constant values to avoid division by zero in correlation
        if np.std(feat_values) == 0 or np.std(errors_insp) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_values, errors_insp)[0, 1]

        print(f"{feat_name}: {corr:.4f}")

    # ==========================================
    # 4. Submission Logic
    # ==========================================
    THRESHOLD = 0.18538684421977109

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} meets threshold ({THRESHOLD}). Generating submission..."
        )
        predict(load_cached_data=True)
    else:
        print(
            f"\nMetric {final_metric} does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
