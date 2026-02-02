import torch
import numpy as np
import pandas as pd
import sys
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.trainer import run_training, generate_submission, set_seed
from library.model import HCCRDSModel
from library.data_loader import process_data


def calculate_column_wise_rmsle(preds_log, targets_log):
    """
    Calculates the mean of column-wise RMSLE.
    Inputs are already log1p transformed (from the model and loader).
    RMSLE_col = RMSE(log_preds, log_targets)
    """
    # Mean Squared Error per column
    mse = np.mean((preds_log - targets_log) ** 2, axis=0)
    # Root Mean Squared Error per column (which represents RMSLE in original space)
    rmsle_per_col = np.sqrt(mse)
    # Final metric is the mean of the column-wise metrics
    final_metric = np.mean(rmsle_per_col)
    return final_metric, rmsle_per_col


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Training
    # Run the training pipeline. This handles data loading, model init, training loop,
    # and returns the model with the best validation weights loaded.
    print("Starting Model Training...")
    # We use the default configuration (200 epochs) as it is fast enough for this dataset size
    # and ensures convergence for the wide architecture.
    model = run_training()
    model.to(device)
    model.eval()

    # 3. Validation & Metric Calculation
    print("\nPerforming Validation Assessment...")
    # Load loaders (using cache generated during training)
    # We only need the val_loader here
    _, val_loader, _ = process_data(load_cached_data=True)

    val_preds = []
    val_targets = []
    val_global_feats = []

    with torch.no_grad():
        for batch in val_loader:
            atomic_features = batch["atomic_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            global_features = batch["global_features"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(atomic_features, batch_indices, global_features)

            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
            # Store global features for failure analysis
            val_global_feats.append(batch["global_features"].numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)
    val_global_feats = np.concatenate(val_global_feats, axis=0)

    # Calculate Metric
    final_metric, col_metrics = calculate_column_wise_rmsle(val_preds, val_targets)

    print(
        f"Validation RMSLE per column: Formation Energy={col_metrics[0]:.6f}, Bandgap={col_metrics[1]:.6f}"
    )
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate error magnitude per sample (Mean Squared Error of logs)
    sample_errors = np.mean((val_preds - val_targets) ** 2, axis=1)

    # Feature names corresponding to the global feature vector construction in data_loader.py
    # Order: [a, b, c, alpha, beta, gamma, volume, density, n_atoms, comp_al, comp_ga, comp_in]
    feature_names = [
        "lattice_a",
        "lattice_b",
        "lattice_c",
        "angle_alpha",
        "angle_beta",
        "angle_gamma",
        "volume",
        "density",
        "num_atoms",
        "comp_al",
        "comp_ga",
        "comp_in",
    ]

    # Ensure dimensions match
    if val_global_feats.shape[1] == len(feature_names):
        df_analysis = pd.DataFrame(val_global_feats, columns=feature_names)
        df_analysis["error"] = sample_errors

        # Compute correlations
        correlations = (
            df_analysis.corr()["error"]
            .drop("error")
            .sort_values(ascending=False, key=abs)
        )

        print("Top correlations between Input Features and Model Error:")
        print(correlations.head(5))
    else:
        print(
            f"Warning: Global feature dimension ({val_global_feats.shape[1]}) does not match expected names list ({len(feature_names)})."
        )

    # 5. Submission
    # Threshold defined in task description
    THRESHOLD = 0.05479004207787702

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")
        generate_submission(model, device)
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
