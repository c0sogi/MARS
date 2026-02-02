import os
import sys
import numpy as np
import pandas as pd
import torch

from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders
from library.model import TMIGN
from library.train import run_training, generate_submission


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Training
    # Limiting to 20 epochs for a fast baseline execution as requested.
    # The dataset is small, so this should be sufficient for initial convergence.
    print("\n=== Starting Training Phase ===")
    run_training(
        epochs=20, batch_size=Config.BATCH_SIZE, patience=Config.PATIENCE, debug=False
    )

    # 3. Validation & Metric Calculation
    print("\n=== Starting Validation & Failure Analysis ===")

    # Load the best model
    model = TMIGN().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print("Critical Error: Best model checkpoint not found.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Get Validation Loader
    # We ignore the train loader here
    _, val_loader = get_dataloaders(batch_size=Config.BATCH_SIZE)

    all_true = []
    all_pred = []
    all_sigma = []

    # Inference Loop
    with torch.no_grad():
        for batch in val_loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)

            target_fvc = batch["fvc"].to(device)
            weeks = batch["weeks"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            base_week = batch["base_week"].to(device)

            # Forward pass
            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

            # Trajectory Calculation
            dt = weeks - base_week
            pred_fvc = base_fvc + alpha * dt
            pred_sigma = sigma_base + sigma_growth * torch.abs(dt)

            # Collect results
            all_true.extend(target_fvc.cpu().numpy())
            all_pred.extend(pred_fvc.cpu().numpy())
            all_sigma.extend(pred_sigma.cpu().numpy())

    # Convert to numpy
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    sigma = np.array(all_sigma)

    # Calculate Metric
    final_metric = calculate_metric(y_true, y_pred, sigma)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Calculate absolute errors
    abs_errors = np.abs(y_true - y_pred)

    # Access the dataframe from the dataset to get features
    # OSICDataset stores the processed dataframe in .df
    val_df = val_loader.dataset.df

    # Features to analyze
    features_to_check = ["Age", "Percent", "Weeks", "Baseline_FVC"]

    print("Correlation between Absolute Error and Input Features:")
    for feat in features_to_check:
        if feat in val_df.columns:
            feat_values = val_df[feat].values

            # Ensure lengths match (sanity check)
            if len(feat_values) == len(abs_errors):
                # Compute Pearson correlation
                corr_matrix = np.corrcoef(feat_values, abs_errors)
                corr = corr_matrix[0, 1]
                print(f"  {feat}: {corr:.4f}")
            else:
                print(
                    f"  {feat}: Dimension mismatch (Data: {len(feat_values)}, Errors: {len(abs_errors)})"
                )
        else:
            print(f"  {feat}: Not found in validation metadata")

    # 5. Submission
    threshold = -6.510164260864258
    print(f"\nMetric Threshold: {threshold}")

    if final_metric > threshold:
        print("Metric check passed. Generating submission...")
        generate_submission(batch_size=Config.BATCH_SIZE)
    else:
        print("Metric check failed. Skipping submission generation.")


if __name__ == "__main__":
    main()
