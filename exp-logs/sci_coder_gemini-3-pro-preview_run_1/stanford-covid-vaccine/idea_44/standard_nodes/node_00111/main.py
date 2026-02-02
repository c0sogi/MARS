import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Ensure local library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, mcrmse
from library.dataset import load_data
from library.model import DualStreamBiGRU
from library.engine import run_training, generate_submission


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Training
    # We run with debug=False to use the full dataset (approx 2k samples).
    # Given the A100 GPU and small dataset size, 20 epochs will complete very quickly.
    print("Starting training process...")
    best_model_path = run_training(debug=False)

    # 3. Validation & Failure Analysis
    print("\nRunning Validation and Failure Analysis...")

    device = torch.device(Config.DEVICE)

    # Load Validation Data
    val_dataset = load_data("val", load_cached_data=True, debug=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Best Model
    model = DualStreamBiGRU()
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # Inference on Validation Set
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            inputs = {"seq": seq, "loop": loop, "dist": dist}
            outputs = model(inputs)  # Shape: (B, 107, 3)

            # Slice to scored positions (first 68)
            outputs_masked = outputs[:, : Config.PRED_LEN, :]
            targets_masked = targets[:, : Config.PRED_LEN, :]

            all_preds.append(outputs_masked.cpu().numpy())
            all_targets.append(targets_masked.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate results
    y_pred = np.concatenate(all_preds, axis=0)  # (N, 68, 3)
    y_true = np.concatenate(all_targets, axis=0)  # (N, 68, 3)

    # Compute Final Metric (MCRMSE)
    final_metric = mcrmse(y_true, y_pred)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate RMSE per sample to correlate with metadata
    # Mean over positions (axis 1) and targets (axis 2)
    sample_mse = np.mean((y_true - y_pred) ** 2, axis=(1, 2))
    sample_rmse = np.sqrt(sample_mse)

    # Load Metadata
    df_val = pd.read_parquet(Config.VAL_PATH)

    # Create Analysis DataFrame
    analysis_df = pd.DataFrame({"id": all_ids, "error_rmse": sample_rmse})

    # Merge with original metadata to get features like signal_to_noise and sequence
    analysis_df = analysis_df.merge(df_val, on="id", how="left")

    # Derive simple features for correlation
    analysis_df["len_A"] = analysis_df["sequence"].apply(lambda x: x.count("A"))
    analysis_df["len_G"] = analysis_df["sequence"].apply(lambda x: x.count("G"))
    analysis_df["len_C"] = analysis_df["sequence"].apply(lambda x: x.count("C"))
    analysis_df["len_U"] = analysis_df["sequence"].apply(lambda x: x.count("U"))

    print("\nFailure Analysis - Correlation with Error (RMSE):")
    cols_to_check = ["signal_to_noise", "SN_filter", "len_A", "len_G", "len_C", "len_U"]

    for col in cols_to_check:
        if col in analysis_df.columns:
            # Skip if column is not numeric
            if not pd.api.types.is_numeric_dtype(analysis_df[col]):
                continue

            corr = analysis_df["error_rmse"].corr(analysis_df[col])
            print(f"  {col}: {corr:.4f}")

    # 4. Submission Logic
    THRESHOLD = 0.6176461577
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(best_model_path, debug=False)
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
