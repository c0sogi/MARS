import sys
import os
import numpy as np
import pandas as pd
import torch

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.train import run_training, validate, set_seed
from library.utils import calc_mcc
from library.dataset import get_dataloaders
from library.inference import predict


def main():
    # --- 1. Configuration ---
    # Override Config for a fast but effective baseline execution
    # We reduce epochs to 5 to meet the time constraint while using the full dataset
    Config.EPOCHS = 5
    Config.BATCH_SIZE = 4096

    print("Configuration configured for fast baseline execution.")

    # --- 2. Training ---
    print("Starting training pipeline...")
    # run_training returns the best model (loaded with weights) and threshold
    # debug=False ensures we train on the full dataset for maximum performance
    # load_cached_data=True utilizes pre-processed parquet files if available
    model, best_threshold = run_training(debug=False, load_cached_data=True)

    # --- 3. Validation & Metrics ---
    print("\nRunning final validation evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Retrieve dataloaders (Validation only needed here)
    # We reload to ensure we have access to the data for failure analysis
    _, val_loader, _ = get_dataloaders(load_cached_data=True, debug=False)

    # Generate probabilities on validation set
    val_probs, val_true = validate(model, val_loader, device)

    # Apply the optimized threshold to get binary predictions
    val_preds = (val_probs >= best_threshold).astype(int)

    # Compute Metric
    final_mcc = calc_mcc(val_true, val_preds)

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {final_mcc}")

    # --- 4. Failure Analysis ---
    print("\nPerforming failure analysis...")

    # Calculate error magnitude
    errors = np.abs(val_true - val_probs)

    # Collect features from validation set to correlate with errors
    feature_batches = []
    condition_batches = []

    # Iterate loader to retrieve input features corresponding to the predictions
    with torch.no_grad():
        for inputs, _ in val_loader:
            feature_batches.append(inputs.cpu().numpy())
            # Reconstruct condition from features if needed, or just skip condition analysis
            # For K-MLP simplified pipeline, we didn't pass condition separately.
            # We can extract is_ground from the features if we knew the index, but for now we skip condition correlation.

    X_features = np.concatenate(feature_batches, axis=0)
    # conditions = np.concatenate(condition_batches, axis=0) # Skipped in simplified analysis

    # Construct feature names for reporting
    # Structure: [Lag_-5 ... Lag_0 ... Lag_5]
    # Per Lag: [P1(7), P2(7), Dist(1), LogDist(1), CloseSpd(1)] -> 17 features
    tracking_cols = Config.TRACKING_COLS  # 7 cols
    lags = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)  # -5 to 5

    feature_names = []
    for lag in lags:
        suffix = f"_lag_{lag}"
        feature_names.extend([c + "_1" + suffix for c in tracking_cols])
        feature_names.extend([c + "_2" + suffix for c in tracking_cols])
        feature_names.extend(
            ["distance" + suffix, "log_distance" + suffix, "closing_speed" + suffix]
        )

    correlations = []

    # 1. Correlation with Condition (Is Ground)
    # Skipped in simplified pipeline
    pass

    # 2. Correlation with Wide Features
    # Normalize Error
    e_mean = np.mean(errors)
    e_std = np.std(errors) + 1e-9
    e_norm = (errors - e_mean) / e_std

    # Normalize Features
    X_mean = np.mean(X_features, axis=0)
    X_std = np.std(X_features, axis=0) + 1e-9
    X_norm = (X_features - X_mean) / X_std

    # Vectorized correlation calculation
    cov = np.mean(X_norm * e_norm[:, None], axis=0)

    for i, name in enumerate(feature_names):
        if i < len(cov):
            correlations.append((name, cov[i]))

    # Filter NaNs and sort by absolute correlation
    clean_corrs = [(n, c) for n, c in correlations if np.isfinite(c)]
    clean_corrs.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for name, corr in clean_corrs[:5]:
        print(f"  {name}: {corr:.4f}")

    # --- 5. Submission ---
    TARGET_METRIC = 0.62458462731896

    if final_mcc > TARGET_METRIC:
        print(
            f"\nValidation Metric ({final_mcc:.6f}) meets threshold ({TARGET_METRIC}). Generating submission..."
        )
        predict(threshold=best_threshold, debug=False, load_cached_data=True)
    else:
        print(
            f"\nValidation Metric ({final_mcc:.6f}) does NOT meet threshold ({TARGET_METRIC}). Submission skipped."
        )


if __name__ == "__main__":
    main()
