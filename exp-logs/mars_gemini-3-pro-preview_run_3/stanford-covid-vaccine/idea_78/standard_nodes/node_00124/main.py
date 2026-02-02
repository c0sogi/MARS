import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.dataset import RNADataset
from library.model import DeepResidualBiGRU
from library.train import run_training, set_seed
from library.metrics import compute_scored_mcrmse


def main():
    # =========================================================================
    # 1. Setup & Training
    # =========================================================================
    # Set seeds for reproducibility
    set_seed(Config.SEED)

    print("Starting Fast Baseline Training...")
    # Run training for limited epochs to ensure fast execution
    # The run_training function handles data loading, model init, and saving the best model
    run_training(epochs=15, debug=False)

    # =========================================================================
    # 2. Validation Inference
    # =========================================================================
    print("Loading best model for validation...")
    device = torch.device(Config.DEVICE)

    # Initialize model and load best weights
    model = DeepResidualBiGRU().to(device)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Load Validation Data
    val_dataset = RNADataset(split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    all_preds = []
    all_targets = []
    all_ids = []

    print("Running validation inference...")
    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            adjacency = batch["adjacency"].to(device)
            bpp_mask = batch["bpp_mask"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            # Forward pass
            outputs = model(features, adjacency, bpp_mask)

            # Collect results (move to CPU to save GPU memory)
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(ids)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # =========================================================================
    # 3. Metric Calculation
    # =========================================================================
    # Compute the official metric
    final_metric = compute_scored_mcrmse(all_preds, all_targets)

    # Print strictly required format
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 4. Failure Analysis
    # =========================================================================
    print("\nPerforming Failure Analysis...")

    # Load metadata to get signal_to_noise and SN_filter
    # We use pandas to read the parquet file directly
    val_df = pd.read_parquet(Config.VAL_DATA_PATH)

    # Ensure alignment between dataset and dataframe
    # RNADataset loads sequentially from the source/cache, so order should match
    if val_df["id"].tolist() != all_ids:
        print("Warning: ID mismatch between dataset and metadata. Re-aligning...")
        val_df = val_df.set_index("id").loc[all_ids].reset_index()

    # Calculate RMSE per sample for the scored columns
    # Slice predictions and targets to seq_scored (68)
    preds_sliced = all_preds[:, : Config.SEQ_SCORED, :].numpy()
    targets_sliced = all_targets[:, : Config.SEQ_SCORED, :].numpy()

    # Identify scored column indices
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_TARGETS
    ]

    # Filter to scored columns
    preds_filtered = preds_sliced[:, :, scored_indices]
    targets_filtered = targets_sliced[:, :, scored_indices]

    # Calculate Mean Squared Error per sample (average over seq_len and targets)
    mse_per_sample = np.mean((preds_filtered - targets_filtered) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    val_df["error_rmse"] = rmse_per_sample

    # Correlation with Signal to Noise
    if "signal_to_noise" in val_df.columns:
        corr_sn = val_df["error_rmse"].corr(val_df["signal_to_noise"])
        print(f"Correlation between Error (RMSE) and Signal_to_Noise: {corr_sn:.4f}")

    # Analysis by SN_filter
    if "SN_filter" in val_df.columns:
        avg_err_pass = val_df[val_df["SN_filter"] == 1]["error_rmse"].mean()
        avg_err_fail = val_df[val_df["SN_filter"] == 0]["error_rmse"].mean()
        print(f"Average Error for High Quality (SN_filter=1): {avg_err_pass:.4f}")
        print(f"Average Error for Low Quality  (SN_filter=0): {avg_err_fail:.4f}")

    # =========================================================================
    # 5. Submission Generation
    # =========================================================================
    THRESHOLD = 0.5884495377540588

    if final_metric < THRESHOLD:
        print("\nMetric condition met. Generating submission...")

        # Load Test Data
        test_dataset = RNADataset(split="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(device)
                adjacency = batch["adjacency"].to(device)
                bpp_mask = batch["bpp_mask"].to(device)
                ids = batch["id"]

                # Forward pass - Output shape (B, 107, 5)
                outputs = model(features, adjacency, bpp_mask)

                test_preds.append(outputs.cpu().numpy())
                test_ids.extend(ids)

        # Concatenate predictions
        test_preds = np.concatenate(test_preds, axis=0)  # (N_test, 107, 5)

        # Prepare Submission DataFrame
        # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        submission_rows = []

        for i, sample_id in enumerate(test_ids):
            sample_pred = test_preds[i]  # Shape (107, 5)
            for seqpos in range(Config.SEQ_LEN):
                row_id = f"{sample_id}_{seqpos}"
                # Get predictions for this position
                row_values = sample_pred[seqpos].tolist()
                submission_rows.append([row_id] + row_values)

        columns = ["id_seqpos"] + Config.TARGET_COLS
        sub_df = pd.DataFrame(submission_rows, columns=columns)

        # Save to Config path (Working Directory)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

        # Save to strictly requested path ./submission/submission.csv
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        specific_sub_path = os.path.join(submission_dir, "submission.csv")
        sub_df.to_csv(specific_sub_path, index=False)
        print(f"Submission also saved to {specific_sub_path}")

    else:
        print(
            f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
