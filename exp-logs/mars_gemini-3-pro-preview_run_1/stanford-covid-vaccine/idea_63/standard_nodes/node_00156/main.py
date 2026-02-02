import sys
import os
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.train import train_model, generate_submission_file
from library.data import get_dataloaders
from library.model import RNAModel
from library.utils import seed_everything, mcrmse_loss


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    print("Starting runfile.py execution...")

    # 2. Train
    # Using full dataset and 20 epochs as per Config/Idea to ensure performance.
    # The dataset is small enough that this fits comfortably within the time limit on GPU.
    best_model_path = train_model(epochs=Config.EPOCHS, max_samples=None)

    # 3. Validation & Metrics
    print("Performing final validation...")
    device = torch.device(Config.DEVICE)

    # Load Validation Data
    _, val_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # Load the best model
    model = RNAModel().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Run Inference on Validation Set
    all_preds = []
    all_targets = []

    print("Running validation inference...")
    with torch.no_grad():
        for seq, loop, dist, tgt in val_loader:
            seq = seq.to(device)
            loop = loop.to(device)
            dist = dist.to(device)

            # Forward pass
            pred = model(seq, loop, dist)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(tgt.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Metric
    # mcrmse_loss expects (N, 107, 3) and handles slicing (first 68) internally
    val_metric = mcrmse_loss(all_preds, all_targets)
    print(f"Final Validation Metric: {val_metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate per-sample error (MCRMSE for that sample)
    # Slice to scored region (first 68)
    pred_scored = all_preds[:, : Config.PRED_LEN, :]
    true_scored = all_targets[:, : Config.PRED_LEN, :]

    # Error per sample: Mean of RMSEs of the 3 columns
    # 1. MSE per column per sample: (N, 3)
    mse_per_sample_col = np.mean((pred_scored - true_scored) ** 2, axis=1)
    # 2. RMSE per column per sample: (N, 3)
    rmse_per_sample_col = np.sqrt(mse_per_sample_col)
    # 3. Mean across columns: (N,)
    sample_errors = np.mean(rmse_per_sample_col, axis=1)

    # Load metadata to correlate errors with features
    val_df = pd.read_parquet(Config.VAL_PARQUET)

    # Ensure alignment (DataLoader should preserve order if shuffle=False)
    if len(val_df) != len(sample_errors):
        print(
            f"Warning: Mismatch in validation set size. DF: {len(val_df)}, Preds: {len(sample_errors)}"
        )
        min_len = min(len(val_df), len(sample_errors))
        val_df = val_df.iloc[:min_len]
        sample_errors = sample_errors[:min_len]

    val_df["error_metric"] = sample_errors

    # Feature Engineering for Analysis
    val_df["len_A"] = val_df["sequence"].apply(lambda x: x.count("A"))
    val_df["len_G"] = val_df["sequence"].apply(lambda x: x.count("G"))
    val_df["len_C"] = val_df["sequence"].apply(lambda x: x.count("C"))
    val_df["len_U"] = val_df["sequence"].apply(lambda x: x.count("U"))

    features = ["signal_to_noise", "SN_filter", "len_A", "len_G", "len_C", "len_U"]

    print("Correlation between Error and Features:")
    for feat in features:
        if feat in val_df.columns:
            # Handle potential NaNs
            valid_data = val_df[[feat, "error_metric"]].dropna()
            if len(valid_data) > 1:
                corr, _ = pearsonr(valid_data[feat], valid_data["error_metric"])
                print(f"  {feat}: {corr:.4f}")

    # 5. Submission
    threshold = 0.6176461577
    if val_metric < threshold:
        print(f"\nMetric {val_metric} < {threshold}. Generating submission...")
        generate_submission_file(best_model_path, max_samples=None)
    else:
        print(f"\nMetric {val_metric} >= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
