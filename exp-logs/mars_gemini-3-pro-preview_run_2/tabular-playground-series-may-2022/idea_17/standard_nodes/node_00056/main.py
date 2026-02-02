import os
import sys
import numpy as np
import pandas as pd
import torch

from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders, process_data
from library.model import HybridTransformerResFunnel
from library.train import train, evaluate


def main():
    # 1. Setup
    seed_everything(Config.RANDOM_STATE)
    device = get_device()

    # 2. Fast Baseline Training
    # We limit the data to 100,000 samples and 5 epochs to create a fast baseline
    # as per the requirement to "Make the model training fast".
    print("--- Starting Fast Baseline Training ---")
    train(debug_subset=100000, epochs=5, patience=3)

    # 3. Validation Assessment
    print("\n--- Starting Validation Assessment ---")
    # Load full validation data (debug_subset=None) to ensure metric is computed on the entire hold-out set
    _, val_loader, _ = get_dataloaders(load_cached_data=True, debug_subset=None)

    # Load the best model
    model = HybridTransformerResFunnel().to(device)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Run inference on validation set
    val_auc, val_preds = evaluate(model, val_loader, device)

    # Print required metric
    print(f"Final Validation Metric: {val_auc}")

    # 4. Failure Analysis
    print("\n--- Starting Failure Analysis ---")
    # Load raw data arrays to get features for correlation analysis
    # We ignore the other returns using _
    _, _, _, _, _, val_raw, _, val_y, _, _, _, _ = process_data(load_cached_data=True)

    # Ensure dimensions match (val_loader iterates over val_raw/val_y in order)
    # val_preds is (N,), val_y is (N,)

    # Calculate Error Magnitude
    error_magnitude = np.abs(val_y - val_preds)

    # Calculate correlations with continuous features
    # Reconstruct feature names based on FeatureEngineer logic in library/data.py
    continuous_cols = [f"f_{i:02d}" for i in range(31) if i != 27]

    correlations = []
    for idx, col_name in enumerate(continuous_cols):
        feature_values = val_raw[:, idx]
        # Handle potential NaNs or constant values if any
        if np.std(feature_values) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feature_values, error_magnitude)[0, 1]
        correlations.append((col_name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.6f}")

    # 5. Submission Generation
    THRESHOLD = 0.9970005855169476

    if val_auc > THRESHOLD:
        print(
            f"\nValidation Metric ({val_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Load full test loader
        _, _, test_loader = get_dataloaders(load_cached_data=True, debug_subset=None)

        # Inference
        _, test_preds = evaluate(model, test_loader, device)

        # Prepare submission dataframe
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
        sample_sub["target"] = test_preds

        # Define output path
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        # Save
        sample_sub.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nValidation Metric ({val_auc}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
