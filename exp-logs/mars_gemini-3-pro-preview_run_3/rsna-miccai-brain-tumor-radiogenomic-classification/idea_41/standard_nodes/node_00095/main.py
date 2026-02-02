import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from provided library files
from library.config import (
    MODEL_SAVE_PATH,
    VAL_META_PATH,
    BATCH_SIZE,
    seed_everything,
)
from library.utils import get_device, calculate_roc_auc
from library.data_loader import get_dataloaders
from library.model import BraTSModel
from library.train import run_training
from library.predict import generate_submission


def main():
    # 1. Setup and Training
    seed_everything()
    print("Starting Training Pipeline...")

    # Execute the training loop
    run_training(load_cached_data=True)

    # 2. Final Validation Evaluation
    print("\nStarting Post-Training Evaluation...")
    device = get_device()

    # Load Validation Data
    _, val_loader, _, _ = get_dataloaders(batch_size=BATCH_SIZE, load_cached_data=True)

    # Initialize Model and Load Best Weights
    model = BraTSModel()
    model.to(device)

    if os.path.exists(MODEL_SAVE_PATH):
        state_dict = torch.load(MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("Error: Model file not found. Training may have failed.")
        return

    # Run Inference on Validation Set
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy().flatten())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Calculate and Print Metric
    val_auc = calculate_roc_auc(all_targets, all_probs)
    print(f"Final Validation Metric: {val_auc}")

    # 3. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Load validation metadata to retrieve meta-features
    val_df = pd.read_parquet(VAL_META_PATH)

    # Get IDs from the dataset to ensure alignment
    val_ids = val_loader.dataset.ids

    # Create a DataFrame for analysis
    analysis_df = pd.DataFrame(
        {"BraTS21ID": val_ids, "target": all_targets, "prob": all_probs}
    )

    # Calculate Absolute Error
    analysis_df["error"] = np.abs(analysis_df["target"] - analysis_df["prob"])

    # Calculate Meta-Features (Slice Counts) from metadata paths
    def get_slice_count(paths):
        return len(paths) if paths is not None else 0

    val_df["flair_count"] = val_df["flair_paths"].apply(get_slice_count)
    val_df["t1w_count"] = val_df["t1w_paths"].apply(get_slice_count)
    val_df["t1wce_count"] = val_df["t1wce_paths"].apply(get_slice_count)
    val_df["t2w_count"] = val_df["t2w_paths"].apply(get_slice_count)

    # Merge analysis data with metadata features
    merged_df = pd.merge(analysis_df, val_df, on="BraTS21ID")

    # Calculate Correlations
    features_to_correlate = ["flair_count", "t1w_count", "t1wce_count", "t2w_count"]

    print("Correlation between Error Magnitude and Input Features:")
    for feat in features_to_correlate:
        if feat in merged_df.columns:
            if merged_df[feat].std() > 0:
                corr, _ = pearsonr(merged_df["error"], merged_df[feat])
                print(f"{feat}: {corr:.4f}")
            else:
                print(f"{feat}: NaN (Constant value)")

    # 4. Submission Generation
    # Threshold defined in the task requirements
    THRESHOLD = 0.6978181818181817

    if val_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({val_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        print(
            f"\nValidation AUC ({val_auc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
