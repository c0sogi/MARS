import os
import sys
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_all, mcrmse_metric
from library.data import get_dataloaders
from library.model import HighCapacityBiGRU, train_model, predict


def calculate_correlation(x, y):
    """Robust correlation calculation using numpy."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    # Drop NaNs if any
    valid_mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid_mask) < 2:
        return 0.0
    return np.corrcoef(x[valid_mask], y[valid_mask])[0, 1]


def main():
    # 1. Setup and Reproducibility
    seed_all(Config.SEED)

    # Override Config for Fast Baseline
    # Limiting epochs to 15 ensures the run completes quickly (well within 1 hour)
    # while providing enough steps for the HighCapacityBiGRU to converge.
    Config.NUM_EPOCHS = 15

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Starting execution with Device: {Config.DEVICE}")
    print(f"Training for {Config.NUM_EPOCHS} epochs...")

    # 2. Data Loading
    # Load cached data if available for speed
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Training
    model = HighCapacityBiGRU()
    # train_model handles moving to device, optimizer, scheduler, and saving best model
    train_model(model, train_loader, val_loader)

    # 4. Validation & Metric Assessment
    print("\nPerforming Final Validation...")
    # Load the best model saved during training
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )
    model.to(Config.DEVICE)
    model.eval()

    # Generate predictions on the full validation set
    # predict returns numpy array (N, L, 5)
    val_preds = predict(model, val_loader, Config.DEVICE)

    # Retrieve ground truth from dataset (N, L, 5)
    val_targets = val_loader.dataset.targets

    # Calculate MCRMSE
    # mcrmse_metric handles slicing to SEQ_SCORED and filtering SCORED_TARGETS
    val_score = mcrmse_metric(val_targets, val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_score}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis on Validation Set...")

    # Load validation metadata to access features like signal_to_noise
    val_meta = pd.read_parquet(Config.VAL_DATA_PATH)

    # Calculate error per sample for correlation analysis
    # We replicate the metric logic: slice to scored positions and columns
    scored_len = Config.SEQ_SCORED
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_TARGETS
    ]

    # Slice targets and preds
    y_true_sliced = val_targets[:, :scored_len, scored_indices]
    y_pred_sliced = val_preds[:, :scored_len, scored_indices]

    # Calculate RMSE per sample (averaged across scored columns)
    # (N, L, C) -> (N, C) [MSE per col] -> (N, C) [RMSE per col] -> (N,) [Mean RMSE]
    mse_per_sample_col = np.mean((y_true_sliced - y_pred_sliced) ** 2, axis=1)
    rmse_per_sample_col = np.sqrt(mse_per_sample_col)
    sample_errors = np.mean(rmse_per_sample_col, axis=1)

    # Add error to metadata
    val_meta["model_error"] = sample_errors

    # Compute Correlations
    print("Correlations between Model Error and Input Features:")

    # 1. Signal to Noise
    if "signal_to_noise" in val_meta.columns:
        corr = calculate_correlation(
            val_meta["signal_to_noise"], val_meta["model_error"]
        )
        print(f"  Error vs Signal_to_Noise: {corr:.6f}")

    # 2. SN Filter
    if "SN_filter" in val_meta.columns:
        corr = calculate_correlation(val_meta["SN_filter"], val_meta["model_error"])
        print(f"  Error vs SN_filter:       {corr:.6f}")

    # 3. Sequence Composition (Nucleotide percentages)
    for nuc in ["A", "G", "U", "C"]:
        col_name = f"pct_{nuc}"
        val_meta[col_name] = val_meta["sequence"].apply(lambda s: s.count(nuc) / len(s))
        corr = calculate_correlation(val_meta[col_name], val_meta["model_error"])
        print(f"  Error vs %{nuc}:              {corr:.6f}")

    # 6. Submission Generation
    THRESHOLD = 0.5884495377540588

    if val_score < THRESHOLD:
        print(
            f"\nValidation score ({val_score}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        test_preds = predict(model, test_loader, Config.DEVICE)

        # Format Submission
        # We need to flatten predictions and create 'id_seqpos' identifiers
        test_ids = test_loader.dataset.ids
        seq_len = Config.SEQ_LEN

        # Flatten preds: (N, L, 5) -> (N*L, 5)
        preds_flat = test_preds.reshape(-1, 5)

        # Generate IDs
        id_seqpos_list = []
        for sample_id in test_ids:
            for pos in range(seq_len):
                id_seqpos_list.append(f"{sample_id}_{pos}")

        # Create DataFrame
        submission_df = pd.DataFrame(preds_flat, columns=Config.TARGET_COLS)
        submission_df.insert(0, "id_seqpos", id_seqpos_list)

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {submission_df.shape}")

    else:
        print(
            f"\nValidation score ({val_score}) did NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
