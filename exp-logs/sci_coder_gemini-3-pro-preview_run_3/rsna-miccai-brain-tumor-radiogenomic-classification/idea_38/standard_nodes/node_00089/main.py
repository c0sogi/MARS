import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.train import run_training, validate, set_seed
from library.data import get_dataloader
from library.model import S3DNet


def main():
    # Set seed for reproducibility
    set_seed(Config.SEED)

    print("Initializing training pipeline...")

    # 1. Train the model
    # run_training handles data loading, model init, training loop, and saving best model
    # It uses the settings in Config (15 epochs, batch size 16) which is suitable for a fast baseline
    model = run_training()

    # 2. Final Validation Assessment
    print("\nRunning final validation assessment...")
    device = torch.device(Config.DEVICE)

    # Load validation data (using cache if available for speed)
    val_loader = get_dataloader("val", load_cached_data=True, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()

    # Calculate metric on full validation set
    # validate() returns (avg_loss, auc)
    val_loss, val_auc = validate(model, val_loader, criterion, device)

    # Print the required metric string without formatting
    print(f"Final Validation Metric: {val_auc}")

    # 3. Failure Analysis
    print("\nPerforming failure analysis...")
    model.eval()

    val_ids = []
    val_probs = []
    val_targets = []

    # Collect predictions for individual analysis
    with torch.no_grad():
        for batch in val_loader:
            even = batch["even"].to(device)
            odd = batch["odd"].to(device)
            targets = batch["target"].to(device)
            ids = batch["BraTS21ID"]

            logits = model(even, odd)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            val_ids.extend(ids)
            val_probs.extend(probs)
            val_targets.extend(targets.cpu().numpy())

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame(
        {"BraTS21ID": val_ids, "prob": val_probs, "target": val_targets}
    )

    # Calculate absolute error
    analysis_df["error"] = np.abs(analysis_df["prob"] - analysis_df["target"])

    # Load validation metadata to get input features (slice counts)
    if os.path.exists(Config.VAL_META_PATH):
        meta_df = pd.read_parquet(Config.VAL_META_PATH)

        # Merge predictions with metadata
        # Ensure IDs are strings in both for merging
        analysis_df["BraTS21ID"] = analysis_df["BraTS21ID"].astype(str)
        meta_df["BraTS21ID"] = meta_df["BraTS21ID"].astype(str)

        merged_df = pd.merge(analysis_df, meta_df, on="BraTS21ID", how="left")

        # Calculate correlations between error and slice counts
        modalities = ["flair", "t1w", "t1wce", "t2w"]
        print("Correlation between Absolute Error and Slice Counts:")

        for mod in modalities:
            path_col = f"{mod}_paths"
            count_col = f"{mod}_count"

            # Calculate slice count from the list of paths
            merged_df[count_col] = merged_df[path_col].apply(
                lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
            )

            # Calculate correlation using numpy
            if merged_df[count_col].std() > 0 and merged_df["error"].std() > 0:
                corr = np.corrcoef(merged_df["error"], merged_df[count_col])[0, 1]
                print(f"  {mod}_count: {corr:.6f}")
            else:
                print(f"  {mod}_count: N/A (Constant or Zero Variance)")
    else:
        print("Validation metadata not found, skipping detailed failure analysis.")

    # 4. Submission Generation
    THRESHOLD = 0.6978181818181817

    if val_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Load test data
        test_loader = get_dataloader("test", load_cached_data=True, shuffle=False)

        test_ids = []
        test_probs = []

        # Inference loop
        with torch.no_grad():
            for batch in test_loader:
                even = batch["even"].to(device)
                odd = batch["odd"].to(device)
                ids = batch["BraTS21ID"]

                logits = model(even, odd)
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

                test_ids.extend(ids)
                test_probs.extend(probs)

        # Create submission DataFrame
        submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": test_probs})

        # Convert BraTS21ID to integer as per sample_submission.csv format
        submission_df["BraTS21ID"] = submission_df["BraTS21ID"].astype(int)

        # Save to file
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation AUC ({val_auc}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
