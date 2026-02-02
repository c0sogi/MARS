import sys
import os
import pandas as pd
import numpy as np
import torch
import scipy.stats as stats

# Ensure current directory is in path for library imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, mcrmse
from library.dataset import get_dataloader
from library.model import RNAModel
from library.engine import run_training, predict_and_submit


def perform_failure_analysis(device):
    """
    Performs failure analysis on the validation set using the best saved model.
    Calculates correlations between error magnitude and input features.
    """
    print("\nStarting Failure Analysis...")

    # 1. Load Validation Data
    val_loader = get_dataloader(mode="val", batch_size=Config.BATCH_SIZE, shuffle=False)

    # 2. Load Best Model
    model = RNAModel()
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 3. Inference
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            seq, loop, dist, targets, ids = batch

            seq = seq.to(device)
            loop = loop.to(device)
            dist = dist.to(device)

            # Forward pass
            outputs = model(seq, loop, dist)

            # Slice to scored length (68)
            outputs_sliced = outputs[:, : Config.PRED_LENGTH, :]

            all_preds.append(outputs_sliced.cpu().numpy())
            all_targets.append(targets.numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 4. Calculate Metric (MCRMSE)
    final_metric = mcrmse(all_targets, all_preds)

    # 5. Calculate Per-Sample Error (RMSE averaged over 3 targets)
    # shape: (N, 68, 3) -> (N, 3) MSE -> (N, 3) RMSE -> (N,) Mean RMSE
    mse_per_sample = np.mean((all_targets - all_preds) ** 2, axis=1)
    rmse_per_sample = np.sqrt(mse_per_sample)
    mean_rmse_per_sample = np.mean(rmse_per_sample, axis=1)

    # 6. Load Metadata for Feature Correlation
    df_val = pd.read_parquet(Config.VAL_METADATA_PATH)
    # Align metadata with the order of predictions using IDs
    df_val = df_val.set_index("id").loc[all_ids].reset_index()

    # Extract features for analysis
    analysis_features = {}

    # Signal Quality
    if "signal_to_noise" in df_val.columns:
        analysis_features["signal_to_noise"] = df_val["signal_to_noise"].values
    if "SN_filter" in df_val.columns:
        analysis_features["SN_filter"] = df_val["SN_filter"].values.astype(float)

    # Sequence Composition
    analysis_features["len_A"] = df_val["sequence"].apply(lambda x: x.count("A")).values
    analysis_features["len_G"] = df_val["sequence"].apply(lambda x: x.count("G")).values
    analysis_features["len_C"] = df_val["sequence"].apply(lambda x: x.count("C")).values
    analysis_features["len_U"] = df_val["sequence"].apply(lambda x: x.count("U")).values

    # 7. Compute and Print Correlations
    print("-" * 50)
    print(f"{'Feature':<25} | {'Correlation with Error':<20}")
    print("-" * 50)

    for feat_name, feat_values in analysis_features.items():
        # Pearson correlation
        if len(np.unique(feat_values)) > 1:
            corr, _ = stats.pearsonr(mean_rmse_per_sample, feat_values)
            print(f"{feat_name:<25} | {corr:.4f}")
        else:
            print(f"{feat_name:<25} | N/A (Constant)")

    print("-" * 50)

    return final_metric


def main():
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Configure Training
    print(f"Initializing run with device: {Config.DEVICE}")

    # 1. Run Training
    # This saves the best model to Config.MODEL_PATH
    run_training()

    # 2. Validation & Failure Analysis
    device = torch.device(Config.DEVICE)
    final_metric = perform_failure_analysis(device)

    # 3. Report Metric
    # Strictly formatted output
    print(f"Final Validation Metric: {final_metric}")

    # 4. Conditional Submission
    THRESHOLD = 0.6209375959946717

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit()
    else:
        print(
            f"\nMetric ({final_metric}) did not beat threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
