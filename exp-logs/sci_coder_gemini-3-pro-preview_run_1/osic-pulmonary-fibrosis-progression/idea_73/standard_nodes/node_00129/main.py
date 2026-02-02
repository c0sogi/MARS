import os
import sys
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.train import run_training
from library.inference import run_inference
from library.model import PCCGNet
from library.data import get_dataloaders
from library.utils import seed_everything, calculate_metric


def main():
    # 1. Setup and Configuration Override
    # Limit epochs to ensure execution finishes within 13 minutes
    Config.EPOCHS = 10
    seed_everything(Config.SEED)

    print("=" * 40)
    print("Step 1: Training Model")
    print("=" * 40)

    # Run the training pipeline
    # This will save the best model to Config.CHECKPOINT_DIR/best_model.pth
    run_training()

    print("\n" + "=" * 40)
    print("Step 2: Validation Assessment")
    print("=" * 40)

    # Load Validation Data
    _, val_loader, _ = get_dataloaders()

    # Load Best Model
    device = torch.device(Config.DEVICE)
    model = PCCGNet().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Run Inference on Validation Set
    all_preds = []
    all_targets = []

    # We also need metadata for failure analysis.
    # The loader returns tensors, so we'll align with the dataframe later
    # or extract batch-wise. Aligning with dataframe is safer if order is preserved (shuffle=False).
    # val_loader in library.data has shuffle=False.

    with torch.no_grad():
        for inputs, targets in val_loader:
            # Move inputs to device
            axial = inputs["axial"].to(device)
            coronal = inputs["coronal"].to(device)
            tabular = inputs["tabular"].to(device)
            delta_week = inputs["delta_week"].to(device)
            base_fvc = inputs["base_fvc"].to(device)

            # Forward pass
            preds = model(axial, coronal, tabular, delta_week, base_fvc)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Metric
    # calculate_metric expects [N, 2] preds and [N] targets
    val_metric = calculate_metric(all_preds, all_targets)

    # Print Required Metric Output
    print(f"Final Validation Metric: {val_metric}")

    print("\n" + "=" * 40)
    print("Step 3: Failure Analysis")
    print("=" * 40)

    # Load validation dataframe to get features
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure lengths match
    if len(val_df) != len(all_preds):
        print(
            f"Warning: Validation DF length ({len(val_df)}) != Predictions length ({len(all_preds)})"
        )
        # Truncate to minimum length for analysis if mismatch occurs (shouldn't happen with correct loader)
        min_len = min(len(val_df), len(all_preds))
        val_df = val_df.iloc[:min_len]
        all_preds = all_preds[:min_len]
        all_targets = all_targets[:min_len]

    # Calculate Absolute Error
    pred_fvc = all_preds[:, 0]
    true_fvc = all_targets
    abs_error = np.abs(true_fvc - pred_fvc)

    # Add to dataframe
    analysis_df = val_df.copy()
    analysis_df["Abs_Error"] = abs_error
    analysis_df["Pred_FVC"] = pred_fvc
    analysis_df["Pred_Sigma"] = all_preds[:, 1]

    # Features to correlate
    features = ["Age", "Percent", "Weeks", "Baseline_FVC"]

    print("Correlation between Absolute Error and Features:")
    for feat in features:
        if feat in analysis_df.columns:
            # Handle potential NaNs just in case
            valid_mask = analysis_df[feat].notna() & analysis_df["Abs_Error"].notna()
            if valid_mask.sum() > 1:
                corr, _ = pearsonr(
                    analysis_df.loc[valid_mask, feat],
                    analysis_df.loc[valid_mask, "Abs_Error"],
                )
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Not enough data")
        else:
            print(f"  {feat}: Column not found")

    print("\n" + "=" * 40)
    print("Step 4: Submission Generation")
    print("=" * 40)

    threshold = -6.510164260864258

    if val_metric > threshold:
        print(
            f"Validation metric ({val_metric}) > Threshold ({threshold}). Generating submission..."
        )
        run_inference()
    else:
        print(
            f"Validation metric ({val_metric}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
