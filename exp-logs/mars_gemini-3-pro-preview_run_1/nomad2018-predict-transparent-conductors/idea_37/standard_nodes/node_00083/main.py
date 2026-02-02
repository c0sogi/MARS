import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from library.config import Config
from library.dataset import prepare_data
from library.engine import run_training, generate_submission, set_seed


def main():
    # 1. Setup and Config Override for Fast Baseline
    # We reduce the number of epochs to ensure the run completes quickly within the time limit
    Config.NUM_EPOCHS = 50
    set_seed(Config.SEED)

    # 2. Prepare Data
    # This will load cached data if available or process from scratch
    train_loader, val_loader, test_loader = prepare_data(load_cached_data=True)

    # 3. Train Model
    # The engine handles the training loop, early stopping, and saving the best model
    model = run_training(train_loader, val_loader)

    # 4. Validation & Metric Calculation
    # We need to manually compute the specific column-wise RMSLE metric
    model.eval()
    device = torch.device(Config.DEVICE)

    all_preds = []
    all_targets = []
    all_global_feats = []

    with torch.no_grad():
        for (
            batch_atom_feats,
            batch_indices,
            batch_global_feats,
            batch_targets,
            _,
        ) in val_loader:
            batch_atom_feats = batch_atom_feats.to(device)
            batch_indices = batch_indices.to(device)
            batch_global_feats = batch_global_feats.to(device)

            # Forward pass (output is log1p space)
            preds = model(batch_atom_feats, batch_indices, batch_global_feats)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_targets.numpy())
            all_global_feats.append(batch_global_feats.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_global_feats = np.concatenate(all_global_feats, axis=0)

    # Calculate Column-wise RMSLE
    # Note: The model predicts log(1+y) and targets are log(1+y).
    # RMSLE is sqrt(mean((log(1+y_pred) - log(1+y_true))^2))
    # which corresponds to RMSE in the log-transformed space.
    mse_col1 = mean_squared_error(all_targets[:, 0], all_preds[:, 0])
    mse_col2 = mean_squared_error(all_targets[:, 1], all_preds[:, 1])

    rmsle_col1 = np.sqrt(mse_col1)
    rmsle_col2 = np.sqrt(mse_col2)

    # The competition metric is the mean of the column-wise RMSLEs
    final_metric = (rmsle_col1 + rmsle_col2) / 2.0

    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nFailure Analysis (Correlation with Error Magnitude):")
    # Calculate error magnitude (Mean Absolute Error in log space per sample)
    # We average the error across the two targets for a single "error score" per sample
    errors = np.mean(np.abs(all_preds - all_targets), axis=1)

    # Global feature names corresponding to geometry_processor.py
    feat_names = [
        "lattice_a",
        "lattice_b",
        "lattice_c",
        "angle_alpha",
        "angle_beta",
        "angle_gamma",
        "volume",
        "density",
        "frac_Al",
        "frac_Ga",
        "frac_In",
        "num_atoms",
    ]

    # Create DataFrame for correlation analysis
    analysis_df = pd.DataFrame(all_global_feats, columns=feat_names)
    analysis_df["error"] = errors

    # Compute correlation of features with the error
    correlations = (
        analysis_df.corr()["error"].drop("error").sort_values(key=abs, ascending=False)
    )
    print(correlations)

    # 6. Submission Logic
    # Threshold defined in the task description
    THRESHOLD = 0.05479004207787702

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")
        generate_submission(model, test_loader)
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
