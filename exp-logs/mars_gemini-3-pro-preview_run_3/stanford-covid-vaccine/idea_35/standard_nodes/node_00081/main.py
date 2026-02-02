import os
import sys
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.train_utils import train_model, generate_submission, set_seed
from library.data_utils import get_dataloaders
from library.model import DDCGBiGRU
from library.loss_metric import competition_metric


def main():
    # 1. Configuration for Fast Baseline
    # Limit epochs to ensure quick execution while using full data for stability.
    # 20 epochs on ~1700 samples is very fast on GPU.
    Config.EPOCHS = 20

    # 2. Train Model
    # We use debug=False to train on the full dataset.
    train_model(debug=False)

    # 3. Validation & Failure Analysis
    set_seed()
    device = torch.device(Config.DEVICE)

    # Load the best saved model
    model = DDCGBiGRU().to(device)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError("Model file not found. Training failed.")

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Get Validation Data
    # Note: train_model already cached the data, so this load is fast
    _, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug=False,
    )

    # Inference on Validation Set
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            adj_indices = batch["adj_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["targets"]  # Keep on CPU
            ids = batch["id"]

            outputs = model(features, adj_indices, pair_mask)

            all_preds.append(outputs.cpu())
            all_targets.append(targets)
            all_ids.extend(ids)

    global_preds = torch.cat(all_preds, dim=0)
    global_targets = torch.cat(all_targets, dim=0)

    # Calculate Final Metric
    metric = competition_metric(global_preds, global_targets)
    print(f"Final Validation Metric: {metric}")

    # Failure Analysis
    # Calculate MCRMSE per sample
    # 1. Slice to scored length (68)
    preds_sliced = global_preds[:, : Config.PRED_LEN, :]
    targets_sliced = global_targets[:, : Config.PRED_LEN, :]

    # 2. Select scored columns
    preds_scored = preds_sliced[:, :, Config.SCORED_COLS_INDICES]
    targets_scored = targets_sliced[:, :, Config.SCORED_COLS_INDICES]

    # 3. Compute RMSE per column per sample: (N, 68, 3) -> (N, 3)
    mse_per_sample_col = torch.mean((preds_scored - targets_scored) ** 2, dim=1)
    rmse_per_sample_col = torch.sqrt(mse_per_sample_col)

    # 4. Average over columns to get MCRMSE per sample: (N,)
    error_per_sample = torch.mean(rmse_per_sample_col, dim=1).numpy()

    # Load Metadata for correlations
    df_val = pd.read_parquet(Config.VAL_PATH)

    # Ensure IDs match
    error_df = pd.DataFrame({"id": all_ids, "error": error_per_sample})

    analysis_df = pd.merge(df_val, error_df, on="id", how="inner")

    print("Failure Analysis (Correlation with Error):")
    features_to_check = ["signal_to_noise", "SN_filter"]

    for feat in features_to_check:
        if feat in analysis_df.columns:
            valid_data = analysis_df[[feat, "error"]].dropna()
            if len(valid_data) > 1:
                corr, _ = pearsonr(valid_data[feat], valid_data["error"])
                print(f"{feat}: {corr}")
            else:
                print(f"{feat}: Not enough data")
        else:
            print(f"{feat}: Column not found")

    # 4. Submission Generation
    # Threshold check
    THRESHOLD = 0.5978901386
    if metric < THRESHOLD:
        generate_submission(debug=False)
    else:
        print(f"Metric {metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
