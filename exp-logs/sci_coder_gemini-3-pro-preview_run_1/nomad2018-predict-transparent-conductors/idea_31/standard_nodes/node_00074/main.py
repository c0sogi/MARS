import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import sys
import os

# Import library components
from library.config import TRAINING_PARAMS, DEVICE, SEED, WORKING_DIR, SUBMISSION_PATH
from library.engine import run_training, generate_submission, set_seed
from library.data import get_data_loaders


def calculate_column_wise_rmsle(preds_log, targets_log):
    """
    Calculates the Column-wise Root Mean Squared Logarithmic Error.
    Since inputs are already in log1p scale, this is the column-wise RMSE.
    """
    # squared errors
    squared_errors = (preds_log - targets_log) ** 2
    # mean squared error per column
    mse_col = np.mean(squared_errors, axis=0)
    # root mean squared error per column
    rmse_col = np.sqrt(mse_col)
    # mean of column-wise RMSEs
    return np.mean(rmse_col)


def main():
    # 1. Setup
    set_seed(SEED)

    # Adjust training params for a fast baseline
    # 100 epochs is sufficient for convergence on this dataset size (~1.7k samples)
    TRAINING_PARAMS["epochs"] = 100

    print(f"Running on device: {DEVICE}")

    # 2. Train Model
    # run_training loads data, trains, handles early stopping, returns best model and test_loader
    print("Starting training process...")
    model, test_loader = run_training(load_cached_data=True)

    # 3. Validation & Metric Calculation
    print("Performing validation inference...")
    # We need the validation loader. We can get it from get_data_loaders.
    # Note: run_training already called get_data_loaders, so data is cached.
    _, val_loader, _ = get_data_loaders(
        batch_size=TRAINING_PARAMS["batch_size"], load_cached_data=True
    )

    model.eval()
    all_preds_log = []
    all_targets_log = []
    all_global_feats = []

    with torch.no_grad():
        for batch in val_loader:
            atomic_feats = batch["atomic_features"].to(DEVICE)
            global_feats = batch["global_features"].to(DEVICE)
            batch_indices = batch["batch_indices"].to(DEVICE)
            targets = batch["targets"].to(DEVICE)

            outputs = model(atomic_feats, global_feats, batch_indices)

            all_preds_log.append(outputs.cpu().numpy())
            all_targets_log.append(targets.cpu().numpy())
            all_global_feats.append(global_feats.cpu().numpy())

    all_preds_log = np.concatenate(all_preds_log, axis=0)
    all_targets_log = np.concatenate(all_targets_log, axis=0)
    all_global_feats = np.concatenate(all_global_feats, axis=0)

    # Calculate Metric
    # The metric is Column-wise Root Mean Squared Logarithmic Error.
    # Our model predicts log(1+y), and targets are log(1+y).
    # So we calculate RMSE on these log values.

    # Global RMSLE (standard RMSE on log data)
    global_mse = mean_squared_error(all_targets_log, all_preds_log)
    global_rmsle = np.sqrt(global_mse)

    # Column-wise RMSLE (average of RMSLE for each target)
    col_wise_rmsle = calculate_column_wise_rmsle(all_preds_log, all_targets_log)

    # The prompt asks for "Column-wise root mean squared logarithmic error".
    # We will print the column-wise version as the primary metric,
    # but check the global one as well.
    print(f"Final Validation Metric: {col_wise_rmsle}")
    # print(f"Global RMSLE: {global_rmsle}")

    # 4. Failure Analysis
    print("\nPerforming failure analysis...")
    # Calculate error magnitude per sample (average L1 error on log scale)
    # error = mean(|pred_log - target_log|) across the two targets
    errors = np.mean(np.abs(all_preds_log - all_targets_log), axis=1)

    # Global feature names based on library.features.get_global_features
    feature_names = [
        "lattice_len_a",
        "lattice_len_b",
        "lattice_len_c",
        "angle_alpha",
        "angle_beta",
        "angle_gamma",
        "volume",
        "density",
        "frac_Al",
        "frac_Ga",
        "frac_In",
        "total_atoms",
    ]

    print("Correlation between Error Magnitude and Global Features:")
    correlations = []
    for i, name in enumerate(feature_names):
        if i < all_global_feats.shape[1]:
            feat_values = all_global_feats[:, i]
            # Handle constant features (std ~ 0) to avoid NaN
            if np.std(feat_values) > 1e-9:
                corr = np.corrcoef(errors, feat_values)[0, 1]
                correlations.append((name, corr))
            else:
                correlations.append((name, 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, corr in correlations:
        print(f"  {name:<15}: {corr:.4f}")

    # 5. Submission
    # Threshold from instructions
    THRESHOLD = 0.05479004207787702

    # Use the calculated metric for decision
    if col_wise_rmsle < THRESHOLD:
        print(
            f"\nValidation metric {col_wise_rmsle} < threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(model, test_loader)
    else:
        print(
            f"\nValidation metric {col_wise_rmsle} >= threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
