import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import sys
import os

# Import from the provided library files
from library.config import SEED
from library.data import get_dataloaders
from library.engine import Engine, set_seed
from library.utils import inverse_log_transform_targets, calculate_rmsle


def perform_failure_analysis(val_globals, val_targets_orig, val_preds_orig):
    """
    Analyzes correlations between global features and prediction errors.
    """
    print("\nFailure Analysis (Correlation with Error Magnitude):")

    # Calculate error magnitude per sample
    # Using Mean Absolute Error on Log Scale as it aligns with RMSLE
    log_preds = np.log1p(val_preds_orig)
    log_targets = np.log1p(val_targets_orig)
    # Average error across the two targets for each sample
    errors = np.mean(np.abs(log_preds - log_targets), axis=1)

    # Feature names corresponding to the order in library.features.process_geometry
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
        "frac_al",
        "frac_ga",
        "frac_in",
        "mean_mass",
        "mean_radius",
        "mean_eneg",
    ]

    correlations = []
    for i, name in enumerate(feature_names):
        if i < val_globals.shape[1]:
            feat_values = val_globals[:, i]
            # Handle constant features to avoid warnings/NaNs
            if np.std(feat_values) > 1e-9:
                corr, _ = pearsonr(feat_values, errors)
                correlations.append((name, corr))
                print(f"{name:<15}: {corr:.4f}")
            else:
                print(f"{name:<15}: NaN (Constant)")

    # Print top contributing feature
    if correlations:
        best_feat, best_corr = max(correlations, key=lambda x: abs(x[1]))
        print(f"\nStrongest correlation with error: {best_feat} ({best_corr:.4f})")


def main():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")
    set_seed(SEED)

    # 2. Data Loading
    # Using cached data for speed as per instructions
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model & Engine Initialization
    engine = Engine(device)

    # 4. Training
    # Using 50 epochs with patience of 10 to ensure a fast but effective baseline
    print("Starting Training...")
    engine.run_training(train_loader, val_loader, epochs=50, patience=10)

    # 5. Validation Assessment
    print("\nRunning Final Validation Assessment...")
    # Ensure model is in eval mode
    engine.model.eval()

    val_preds = []
    val_targets = []
    val_globals = []

    with torch.no_grad():
        for batch_atomic, batch_indices, batch_global, batch_targets, _ in val_loader:
            batch_atomic = batch_atomic.to(device)
            batch_indices = batch_indices.to(device)
            batch_global = batch_global.to(device)
            batch_targets = batch_targets.to(device)

            # Forward pass
            outputs = engine.model(batch_atomic, batch_global, batch_indices, None)

            # Collect results
            val_preds.append(outputs.cpu().numpy())
            val_targets.append(batch_targets.cpu().numpy())
            val_globals.append(batch_global.cpu().numpy())

    # Concatenate batches
    val_preds = np.vstack(val_preds)
    val_targets = np.vstack(val_targets)
    val_globals = np.vstack(val_globals)

    # Inverse transform to original scale for metric calculation
    val_preds_orig = inverse_log_transform_targets(val_preds)
    val_targets_orig = inverse_log_transform_targets(val_targets)

    # Compute Metric
    final_metric = calculate_rmsle(val_targets_orig, val_preds_orig)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    perform_failure_analysis(val_globals, val_targets_orig, val_preds_orig)

    # 7. Submission Generation
    THRESHOLD = 0.05479004207787702
    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} meets threshold ({THRESHOLD}). Generating submission..."
        )
        engine.generate_submission(test_loader)
    else:
        print(
            f"\nMetric {final_metric} does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
