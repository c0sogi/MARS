import os
import sys
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.train import train_model
from library.model import SDBR_BiGRU
from library.data import get_dataloaders
from library.utils import seed_everything, compute_mcrmse


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running on device: {device}")

    # 2. Train Model
    # We use the default configuration (25 epochs, full dataset) which is optimized
    # for both speed (on A100) and performance.
    print("Starting training pipeline...")
    best_val_score = train_model()

    # 3. Validation & Failure Analysis
    print("\nRunning Validation and Failure Analysis...")

    # Load best model
    model = SDBR_BiGRU().to(device)
    model_path = Config.MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Training might have failed.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get dataloaders (reuses cache generated during training)
    # We need the validation loader for analysis and test loader for submission
    _, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Validation Inference
    val_preds_list = []
    val_targets_list = []
    val_ids_list = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"].cpu().numpy()
            ids = batch["id"]

            preds = model(features, pair_indices, pair_masks)
            val_preds_list.append(preds.cpu().numpy())
            val_targets_list.append(targets)
            val_ids_list.extend(ids)

    val_preds = np.concatenate(val_preds_list, axis=0)
    val_targets = np.concatenate(val_targets_list, axis=0)

    # Calculate Final Metric (Scored Columns Only)
    # compute_mcrmse handles slicing to seq_scored (68) internally
    final_metric = compute_mcrmse(val_preds, val_targets, scored_only=True)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate RMSE per sample on scored columns
    # Slice predictions to match targets (68 positions)
    seq_scored = val_targets.shape[1]
    val_preds_sliced = val_preds[:, :seq_scored, :]

    # Identify indices of scored columns
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    # Squared error: (N, 68, 5)
    squared_error = (val_preds_sliced - val_targets) ** 2

    # Filter for scored columns -> (N, 68, 3)
    squared_error_scored = squared_error[:, :, scored_indices]

    # Mean over length and columns -> (N,)
    mse_per_sample = np.mean(squared_error_scored, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load metadata to correlate
    val_meta_df = pd.read_parquet(Config.VAL_PATH)

    # Create error dataframe
    error_df = pd.DataFrame({"id": val_ids_list, "rmse": rmse_per_sample})

    # Merge with metadata on 'id'
    analysis_df = pd.merge(error_df, val_meta_df, on="id")

    # Calculate correlations with key metadata
    cols_to_correlate = ["rmse", "signal_to_noise", "SN_filter"]
    # Ensure columns exist (SN_filter might be categorical/int)
    cols_to_correlate = [c for c in cols_to_correlate if c in analysis_df.columns]

    correlations = analysis_df[cols_to_correlate].corr()["rmse"]
    print("\nFailure Analysis - Correlation with Error (RMSE):")
    print(correlations.drop("rmse"))

    # 4. Submission
    THRESHOLD = 0.5884495377540588

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric {final_metric} < {THRESHOLD}. Generating submission..."
        )

        test_preds_list = []
        test_ids_list = []

        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(device)
                pair_indices = batch["pair_indices"].to(device)
                pair_masks = batch["pair_masks"].to(device)
                ids = batch["id"]

                preds = model(features, pair_indices, pair_masks)
                test_preds_list.append(preds.cpu().numpy())
                test_ids_list.extend(ids)

        # (N_test, 107, 5)
        test_preds = np.concatenate(test_preds_list, axis=0)

        # Flatten for submission
        # Format: id_seqpos, then the 5 target columns
        submission_data = []
        target_cols = Config.TARGET_COLS

        for i, sample_id in enumerate(test_ids_list):
            sample_preds = test_preds[i]  # Shape: (107, 5)
            for seqpos in range(Config.SEQ_LEN):
                row_id = f"{sample_id}_{seqpos}"
                row_values = sample_preds[seqpos].tolist()
                submission_data.append([row_id] + row_values)

        columns = ["id_seqpos"] + target_cols
        submission_df = pd.DataFrame(submission_data, columns=columns)

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {submission_df.shape}")

    else:
        print(
            f"\nValidation metric {final_metric} >= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
