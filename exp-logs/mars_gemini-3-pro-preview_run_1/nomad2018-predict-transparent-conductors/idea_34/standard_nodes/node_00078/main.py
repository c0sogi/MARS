import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import provided libraries
from library.config import Config
from library.trainer import Runner
from library.data_factory import DataProcessor


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    # Reduce epochs for fast baseline execution as required
    Config.NUM_EPOCHS = 50

    print("=== Configuration ===")
    Config.print_config()
    print("=====================\n")

    # -------------------------------------------------------------------------
    # 2. Model Training
    # -------------------------------------------------------------------------
    # Initialize Runner
    runner = Runner()

    # Train the model (uses cached data if available, otherwise computes it)
    runner.train(load_cached_data=True)

    # -------------------------------------------------------------------------
    # 3. Validation & Metric Calculation
    # -------------------------------------------------------------------------
    print("\n=== Validation & Failure Analysis ===")

    # Load validation loader
    # We access the processor inside runner to reuse scalers if needed
    _, val_loader, _ = runner.processor.get_dataloaders(load_cached_data=True)

    # Load best model
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Error: Model checkpoint not found at {Config.MODEL_SAVE_PATH}")
        return

    runner.model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=runner.device)
    )
    runner.model.eval()

    all_preds_log = []
    all_targets_log = []
    all_global_feats = []

    # Run inference on validation set (no gradients for speed and memory efficiency)
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            atomic_features = batch["atomic_features"].to(runner.device)
            batch_indices = batch["batch_indices"].to(runner.device)
            global_features = batch["global_features"].to(runner.device)
            targets = batch["targets"].to(runner.device)

            # Forward pass
            outputs = runner.model(atomic_features, batch_indices, global_features)

            # Collect results (keep on CPU as numpy)
            all_preds_log.append(outputs.cpu().numpy())
            all_targets_log.append(targets.cpu().numpy())
            # Collect global features for failure analysis
            all_global_feats.append(batch["global_features"].cpu().numpy())

    # Concatenate all batches
    preds_log = np.concatenate(all_preds_log, axis=0)
    targets_log = np.concatenate(all_targets_log, axis=0)
    global_feats = np.concatenate(all_global_feats, axis=0)

    # Calculate Column-wise RMSLE
    # Note: targets and preds are already in log1p space (from training setup).
    # RMSE(log_pred, log_target) is mathematically equivalent to RMSLE(pred, target)
    mse_per_col = np.mean((targets_log - preds_log) ** 2, axis=0)
    rmse_per_col = np.sqrt(mse_per_col)
    mcrmse = np.mean(rmse_per_col)

    print(f"Final Validation Metric: {mcrmse}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    # Calculate mean absolute error per sample in log space
    sample_errors = np.mean(np.abs(targets_log - preds_log), axis=1)

    # Define feature names for global features (12 dimensions) based on geometry_utils
    # [a, b, c, alpha, beta, gamma, vol, density, frac_Al, frac_Ga, frac_In, n_atoms]
    feature_names = [
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

    # Create DataFrame for analysis
    # Note: global_feats are scaled (z-score), but correlation is scale-invariant
    df_analysis = pd.DataFrame(global_feats, columns=feature_names)
    df_analysis["Error"] = sample_errors

    # Compute correlation
    correlations = (
        df_analysis.corr()["Error"].drop("Error").sort_values(key=abs, ascending=False)
    )

    print("\nCorrelation between Model Error and Global Features:")
    print(correlations)

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.05479004207787702

    if mcrmse < THRESHOLD:
        print(
            f"\nValidation metric {mcrmse} meets threshold {THRESHOLD}. Generating submission..."
        )
        runner.predict(load_cached_data=True)
    else:
        print(
            f"\nValidation metric {mcrmse} does not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
