import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, mcrmse
from library.data import get_dataloaders
from library.model import GSRDN
from library.train import run_training, generate_submission


def main():
    # 1. Configuration and Setup
    config = Config()
    set_seed(config.seed)

    # 2. Training
    # Run training for 15 epochs to ensure a fast baseline execution.
    # The run_training function handles data loading, model init, and saving the best model.
    print("Starting training...")
    run_training(epochs=15, batch_size=32)

    # 3. Validation & Metric Reporting
    print("Loading best model for validation analysis...")
    device = config.device
    model = GSRDN().to(device)

    if not os.path.exists(config.model_save_path):
        raise FileNotFoundError(f"Model file not found at {config.model_save_path}")

    model.load_state_dict(torch.load(config.model_save_path, map_location=device))
    model.eval()

    # Get validation dataloader
    _, val_loader, _ = get_dataloaders()

    all_preds = []
    all_targets = []

    # Indices for scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C
    scored_indices = [config.target_cols.index(col) for col in config.scored_cols]

    # Inference loop on Validation Set
    with torch.no_grad():
        for features, pidx, targets in val_loader:
            features = features.to(device)
            pidx = pidx.to(device)

            # 1. Static Features
            z = model.get_static_features(features)

            # 2. Pass 1 (Zero Feedback)
            preds_1 = model(features, pidx, prev_preds=None, z_cached=z)

            # 3. Pass 2 (Graph-Smoothed Feedback - Final Prediction)
            preds_2 = model(features, pidx, prev_preds=preds_1, z_cached=z)

            # Collect predictions and targets for metric calculation
            # Slice to seq_scored length (68)
            seq_scored = targets.shape[1]
            preds_cpu = preds_2[:, :seq_scored, :].cpu().numpy()
            targets_cpu = targets.cpu().numpy()

            # Extract only the scored columns
            all_preds.append(preds_cpu[:, :, scored_indices])
            all_targets.append(targets_cpu[:, :, scored_indices])

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute Final Metric
    final_metric = mcrmse(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate RMSE per sample (averaging MSE over sequence and channels, then sqrt)
    # Shape: (N_samples,)
    mse_per_sample = np.mean((all_targets - all_preds) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load Validation Metadata to get features
    val_df = pd.read_csv(config.val_metadata_path)

    # Ensure alignment
    if len(rmse_per_sample) != len(val_df):
        # In case dataloader dropped samples (unlikely with default settings but safe to handle)
        val_df = val_df.iloc[: len(rmse_per_sample)]

    val_df["error_magnitude"] = rmse_per_sample

    # Features to analyze
    features_to_analyze = ["signal_to_noise", "mean_reactivity", "SN_filter"]
    available_features = [f for f in features_to_analyze if f in val_df.columns]

    print("Correlation between Error Magnitude and Input Features:")
    for feature in available_features:
        # Filter out NaNs if any
        valid_mask = val_df[feature].notna() & val_df["error_magnitude"].notna()
        if valid_mask.sum() > 1:
            corr, _ = pearsonr(
                val_df.loc[valid_mask, feature],
                val_df.loc[valid_mask, "error_magnitude"],
            )
            print(f"  {feature}: {corr:.4f}")
        else:
            print(f"  {feature}: Insufficient data")

    # 5. Submission Generation
    threshold = 0.47142532743789534

    if final_metric < threshold:
        print(
            f"\nMetric {final_metric} meets threshold {threshold}. Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nMetric {final_metric} does not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
