import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config, set_seed
from library.dataset import load_data
from library.model import DASR_BiGRU
from library.utils import MCRMSELoss, process_submission
from library.train import run_training


def main():
    # ==========================================
    # 1. Setup & Config Overrides
    # ==========================================
    # Limit epochs for a fast baseline execution
    Config.EPOCHS = 20
    Config.BATCH_SIZE = 64

    # Cite debug_lesson_10: Patch Instances Directly to Bypass Stale Module Caching
    Config.SCORED_COLS_INDICES = [0, 1, 3]

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Training
    # ==========================================
    print("Starting training process...")
    # Explicitly pass epochs to override default argument value
    run_training(
        epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # ==========================================
    # 3. Load Best Model
    # ==========================================
    print("Loading best model for analysis...")
    model = DASR_BiGRU().to(device)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # ==========================================
    # 4. Final Validation Assessment
    # ==========================================
    print("Performing final validation assessment...")
    val_dataset = load_data("val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    criterion = MCRMSELoss()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            feat, pair_idx, dist, targets, mask = batch
            feat = feat.to(device)
            pair_idx = pair_idx.to(device)
            dist = dist.to(device)

            # Inference
            preds = model(feat, pair_idx, dist)

            # Move to CPU for aggregation
            all_preds.append(preds.cpu())
            all_targets.append(targets)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Slice to scored positions (first 68 bases)
    # Cite debug_lesson_1: Align Metric Calculation with Scored Targets
    preds_scored = all_preds[:, : Config.SEQ_SCORED, Config.SCORED_COLS_INDICES]
    targets_scored = all_targets[:, : Config.SEQ_SCORED, Config.SCORED_COLS_INDICES]

    # Calculate Global Metric
    final_loss = criterion(preds_scored, targets_scored)
    metric_val = final_loss.item()

    # REQUIRED OUTPUT: Print full precision metric
    print(f"Final Validation Metric: {metric_val}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\nPerforming failure analysis...")

    # Calculate RMSE per sample (averaging over 68 positions and 5 targets)
    # Squared diff: (N, 68, 5)
    squared_diff = (preds_scored - targets_scored) ** 2
    # Mean over seq and targets: (N,)
    mse_per_sample = torch.mean(squared_diff, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # Load metadata to correlate errors with features
    val_df = pd.read_parquet(Config.VAL_METADATA)

    # Create analysis dataframe
    analysis_df = val_df.copy()
    analysis_df["error_rmse"] = rmse_per_sample

    # Feature Engineering for Analysis
    analysis_df["pct_A"] = analysis_df["sequence"].apply(
        lambda s: s.count("A") / len(s)
    )
    analysis_df["pct_G"] = analysis_df["sequence"].apply(
        lambda s: s.count("G") / len(s)
    )
    analysis_df["pct_U"] = analysis_df["sequence"].apply(
        lambda s: s.count("U") / len(s)
    )
    analysis_df["pct_C"] = analysis_df["sequence"].apply(
        lambda s: s.count("C") / len(s)
    )

    # Calculate Correlations
    corr_cols = [
        "error_rmse",
        "signal_to_noise",
        "SN_filter",
        "pct_A",
        "pct_G",
        "pct_U",
        "pct_C",
    ]
    correlations = (
        analysis_df[corr_cols].corr()["error_rmse"].sort_values(ascending=False)
    )

    print("Correlation between Error (RMSE) and features:")
    print(correlations.drop("error_rmse"))

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    THRESHOLD = 0.5978901386

    if metric_val < THRESHOLD:
        print(
            f"\nMetric {metric_val} is better than threshold {THRESHOLD}. Generating submission..."
        )

        test_dataset = load_data("test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        test_preds = []
        test_ids = test_dataset.ids

        with torch.no_grad():
            for batch in test_loader:
                feat, pair_idx, dist = batch
                feat = feat.to(device)
                pair_idx = pair_idx.to(device)
                dist = dist.to(device)

                # Inference on test set
                preds = model(feat, pair_idx, dist)
                test_preds.append(preds.cpu())

        # Concatenate predictions
        test_preds = torch.cat(test_preds, dim=0)

        # Format and Save
        process_submission(test_preds, test_ids, save_path=Config.SUBMISSION_PATH)

    else:
        print(
            f"\nMetric {metric_val} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
