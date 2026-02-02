import sys
import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

# Import from the provided library
from library.config import Config, set_seed
from library.data import get_dataloaders, get_test_dataloader
from library.model import TemporalCNN, train_model, predict_and_submit


def main():
    # 1. Configuration for Fast Baseline
    # We enable DEBUG mode to use a subset of data for quick execution within the time limit.
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 200000  # 200k samples for robust but fast training
    Config.NUM_EPOCHS = 5  # 5 epochs is sufficient for a baseline
    Config.BATCH_SIZE = 512  # Large batch size for A100 GPU efficiency

    # Ensure reproducibility
    set_seed(Config.SEED)

    print("Configuration:")
    print(f"  Debug Mode: {Config.DEBUG}")
    print(f"  Subset Size: {Config.DEBUG_SUBSET_SIZE}")
    print(f"  Epochs: {Config.NUM_EPOCHS}")
    print(f"  Device: {Config.DEVICE}")

    # 2. Data Loading
    print("\nLoading Data...")
    train_loader, val_loader = get_dataloaders()

    # 3. Model Initialization
    print("Initializing Model...")
    model = TemporalCNN()

    # 4. Training
    print("\nStarting Training...")
    # train_model returns the model with the best weights loaded
    best_model = train_model(model, train_loader, val_loader, Config.DEVICE)

    # 5. Validation & Failure Analysis
    print("\nPerforming Validation and Failure Analysis...")
    best_model.eval()

    val_errors = []
    feat_n_pulses = []
    feat_tot_charge = []

    # Disable gradients for inference
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(Config.DEVICE)
            y = y.to(Config.DEVICE)

            # Predict
            pred = best_model(x)

            # --- Metric Calculation (Element-wise) ---
            # Normalize vectors
            pred_norm = F.normalize(pred, p=2, dim=1)
            target_norm = F.normalize(y, p=2, dim=1)

            # Dot product -> Angle
            # Clamp to [-1, 1] to handle numerical instability
            dot_prod = torch.sum(pred_norm * target_norm, dim=1)
            dot_prod = torch.clamp(dot_prod, -1.0, 1.0)
            angles = torch.acos(dot_prod)

            # Store errors
            val_errors.extend(angles.cpu().numpy())

            # --- Feature Extraction for Analysis ---
            # x shape: (Batch, Seq_Len, 5) -> (x, y, z, time, charge)
            # We extract features from the CPU copy of the input tensor
            x_cpu = x.cpu().numpy()

            # Charge is at index 4.
            # Note: Charge is log1p transformed in data.py.
            # 0 means no pulse (padding), >0 means pulse exists.
            charge_channel = x_cpu[:, :, 4]

            # Feature 1: Number of pulses (non-zero entries)
            n_pulses = (charge_channel > 0).sum(axis=1)
            feat_n_pulses.extend(n_pulses)

            # Feature 2: Total Charge (Sum of log-charge as a proxy for signal strength)
            tot_charge = charge_channel.sum(axis=1)
            feat_tot_charge.extend(tot_charge)

    # Compute Final Metric
    final_metric = np.mean(val_errors)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # Compute Correlations
    df_analysis = pd.DataFrame(
        {
            "error": val_errors,
            "n_pulses": feat_n_pulses,
            "total_charge": feat_tot_charge,
        }
    )

    correlations = df_analysis.corr()["error"].drop("error")
    print("\nFailure Analysis - Correlation with Error:")
    print(correlations)

    # 6. Submission
    print("\nGenerating Submission...")
    test_loader = get_test_dataloader()
    predict_and_submit(best_model, test_loader, Config.DEVICE)

    print("\nPipeline Completed Successfully.")


if __name__ == "__main__":
    main()
