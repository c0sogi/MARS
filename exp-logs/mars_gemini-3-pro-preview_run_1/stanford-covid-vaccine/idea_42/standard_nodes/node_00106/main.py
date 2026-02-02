import pandas as pd
import numpy as np
import torch
import os
import sys

# Import from provided library
from library.config import Config
from library.utils import set_seed, mcrmse_loss
from library.engine import train_model, generate_submission
from library.data import get_loader
from library.model import RNAModel


def run_pipeline():
    # 1. Setup
    set_seed()
    device = Config.device
    print(f"Running on device: {device}")

    # 2. Train Model
    # We use the default epochs from Config (20).
    # Given the small dataset size (approx 1700 samples), this is very fast.
    print("Starting training...")
    best_val_score = train_model(
        epochs=Config.epochs, patience=5, load_cached_data=True
    )

    # 3. Validation & Failure Analysis
    print("\nRunning Validation and Failure Analysis...")

    # Load the best model state
    model = RNAModel().to(device)
    model_path = Config.model_save_path
    if not os.path.exists(model_path):
        print("Model file not found. Training might have failed.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get Validation Loader
    val_loader = get_loader("val", shuffle=False, load_cached_data=True)

    all_preds = []
    all_targets = []
    all_ids = []

    # Inference on Validation Set
    with torch.no_grad():
        for batch in val_loader:
            seq = batch["sequence"].to(device)
            loop = batch["loop_type"].to(device)
            pair = batch["pair_offset"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            preds = model(seq, loop, pair)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(ids)

    # Concatenate results
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate Final Metric (MCRMSE)
    # mcrmse_loss handles slicing to Config.pred_len internally
    final_metric = mcrmse_loss(all_targets, all_preds).item()

    # Print exactly as requested
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    # Calculate RMSE per sample for correlation analysis
    # Slice to scored length for error calculation manually here for per-sample logic
    scored_len = Config.pred_len
    preds_scored = all_preds[:, :scored_len, :]
    targets_scored = all_targets[:, :scored_len, :]

    # Mean Squared Error per sample (average over sequence and channels)
    mse_per_sample = torch.mean((targets_scored - preds_scored) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # Load Validation Metadata to get features
    df_val = pd.read_parquet(Config.val_metadata)

    # Map errors to IDs to ensure alignment
    error_map = dict(zip(all_ids, rmse_per_sample))
    df_val["error"] = df_val["id"].map(error_map)

    # Drop rows where error might be NaN (shouldn't happen if ids match)
    df_val = df_val.dropna(subset=["error"])

    # Generate features for correlation
    # 1. Nucleotide Counts
    df_val["len_A"] = df_val["sequence"].apply(lambda x: x.count("A"))
    df_val["len_G"] = df_val["sequence"].apply(lambda x: x.count("G"))
    df_val["len_C"] = df_val["sequence"].apply(lambda x: x.count("C"))
    df_val["len_U"] = df_val["sequence"].apply(lambda x: x.count("U"))

    features_to_analyze = [
        "signal_to_noise",
        "SN_filter",
        "len_A",
        "len_G",
        "len_C",
        "len_U",
    ]

    print("\nCorrelation between Model Error (RMSE) and Features:")
    for feat in features_to_analyze:
        if feat in df_val.columns:
            # Ensure numeric
            if pd.api.types.is_numeric_dtype(df_val[feat]):
                corr = df_val["error"].corr(df_val[feat])
                print(f"  {feat}: {corr:.4f}")

    # 4. Submission Logic
    threshold = 0.6199890971183777
    if final_metric < threshold:
        print(
            f"\nValidation metric {final_metric:.6f} is better than threshold {threshold:.6f}."
        )
        print("Generating submission file...")
        generate_submission(load_cached_data=True)
    else:
        print(
            f"\nValidation metric {final_metric:.6f} did not meet threshold {threshold:.6f}."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    run_pipeline()
