import sys
import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import prepare_datasets
from library.model import train_model, predict_and_submit


def main():
    # 1. Configuration for Full Training
    # Scaling up to full dataset and extended epochs for convergence (Cite solution_lesson_node_00054, solution_lesson_node_00039)
    Config.EPOCHS = 80
    Config.BATCH_SIZE = 128
    Config.DEBUG = False
    Config.DEBUG_SAMPLE_SIZE = 30000  # Ignored when DEBUG=False

    # 2. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")
    print(
        f"Configuration: Epochs={Config.EPOCHS}, Debug={Config.DEBUG}, Sample Size={Config.DEBUG_SAMPLE_SIZE}"
    )

    # 3. Prepare Datasets
    # load_cached_data is handled internally by prepare_datasets logic (default behavior)
    train_loader, val_loader, test_loader = prepare_datasets(
        batch_size=Config.BATCH_SIZE, force_recompute=False
    )

    # 4. Train Model
    print("\n=== Starting Training ===")
    model = train_model(train_loader, val_loader)

    # 5. Validation Assessment & Failure Analysis
    print("\n=== Starting Validation Assessment ===")
    model.eval()

    total_ae_sum = 0.0
    total_count = 0

    # Storage for failure analysis
    # We will store flattened arrays of features and errors for the inspiratory phase
    all_features = []
    all_errors = []

    # Feature names corresponding to the last dimension of input x
    # Must match the order in library/features.py
    feature_cols = [
        "time_step",
        "u_in",
        "u_out",
        "R",
        "C",
        "area",
        "u_in_diff",
        "u_in_next1",
        "u_in_next2",
        "u_in_next3",
        "u_in_next4",
        "u_in_diff_next1",
        "R_u_in",
        "area_C",
    ]

    with torch.no_grad():
        for x, u_out, y, _ in val_loader:
            x = x.to(device)
            u_out = u_out.to(device)
            y = y.to(device)

            # Forward pass
            preds = model(x)

            # Calculate Absolute Error
            error = torch.abs(preds - y)

            # Mask for inspiratory phase (u_out == 0)
            # Using < 0.5 for float robustness
            mask = u_out < 0.5

            # Accumulate Metric
            masked_error = error[mask]
            total_ae_sum += masked_error.sum().item()
            total_count += mask.sum().item()

            # Collect data for failure analysis
            # We only care about errors in the inspiratory phase
            if mask.sum() > 0:
                # Features for these specific time steps
                # x is (Batch, 80, Features) -> mask selects relevant steps -> (N_masked, Features)
                masked_features = x[mask]

                all_errors.append(masked_error.cpu().numpy())
                all_features.append(masked_features.cpu().numpy())

    # Compute Final Metric
    final_metric = total_ae_sum / total_count
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    if len(all_errors) > 0:
        flat_errors = np.concatenate(all_errors)
        flat_features = np.concatenate(all_features, axis=0)

        print(f"Analyzing correlations on {len(flat_errors)} inspiratory time steps.")
        print(f"{'Feature':<20} | {'Correlation with Error':<25}")
        print("-" * 50)

        for i, feature_name in enumerate(feature_cols):
            # Extract column i
            feature_vals = flat_features[:, i]

            # Compute Pearson correlation
            if np.std(feature_vals) == 0 or np.std(flat_errors) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(feature_vals, flat_errors)[0, 1]

            print(f"{feature_name:<20} | {corr:.6f}")
    else:
        print("No inspiratory phase data found for analysis.")

    # 7. Conditional Submission
    THRESHOLD = 0.1642141044139862

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(model, test_loader)
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
