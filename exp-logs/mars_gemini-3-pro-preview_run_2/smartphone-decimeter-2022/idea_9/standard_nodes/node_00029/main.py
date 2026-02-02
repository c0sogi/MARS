import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import configuration and modules
import library.config
import library.trainer
import library.inference
import library.data_loader
import library.preprocessing
import library.model
import library.utils

# ---------------------------------------------------------
# 1. Configuration Override for Fast Baseline
# ---------------------------------------------------------
# We modify the trainer settings to ensure execution finishes quickly.
print("Configuring fast baseline settings...")
library.trainer.NUM_EPOCHS = 10
library.trainer.EARLY_STOPPING_PATIENCE = 3
# Ensure batch size is optimized for A100
library.trainer.BATCH_SIZE = 512

# Import functions after configuration
from library.trainer import train_model
from library.inference import generate_predictions
from library.data_loader import get_dataset
from library.preprocessing import GNSSSequenceDataset
from library.utils import meters_to_degrees, haversine_distance
from library.config import (
    TRAJECTORY_FEATURES,
    CONTEXT_FEATURES,
    BATCH_SIZE,
    NUM_WORKERS,
    WORKING_DIR,
    WINDOW_SIZE,
)


def calculate_metric(df):
    """
    Calculates the mean of the 50th and 95th percentile distance errors
    averaged across phones, as per the competition metric.
    """
    # Calculate distance error in meters
    df["error_m"] = haversine_distance(
        df["LatitudeDegrees"].values,
        df["LongitudeDegrees"].values,
        df["pred_lat"].values,
        df["pred_lon"].values,
    )

    # Calculate metric per trip (phone run)
    score_list = []
    for trip_id, group in df.groupby("tripId"):
        errors = group["error_m"].values
        if len(errors) == 0:
            continue
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        score_list.append((p50 + p95) / 2)

    if not score_list:
        return float("inf")

    final_score = np.mean(score_list)
    return final_score


def run_validation_and_analysis(model, scaler, val_df):
    """
    Runs inference on the validation set, computes the metric, and performs failure analysis.
    """
    print("\nRunning validation inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Create validation dataset
    val_dataset = GNSSSequenceDataset(val_df, scaler, is_test=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    all_preds = []

    # Lists to store features for failure analysis
    feature_data = {"mean_cn0": [], "mean_uncertainty": [], "sv_count": []}

    # Feature indices for extraction
    idx_cn0 = TRAJECTORY_FEATURES.index("mean_cn0")
    idx_unc = TRAJECTORY_FEATURES.index("mean_uncertainty")
    idx_sv = CONTEXT_FEATURES.index("sv_count")

    with torch.no_grad():
        for traj, ctx, _ in val_loader:
            traj = traj.to(device)
            ctx = ctx.to(device)

            # Forward pass
            output = model(traj, ctx)
            preds = output.cpu().numpy()
            all_preds.append(preds)

            # Extract features from the center of the window for analysis
            # traj shape: [Batch, Channels, Length]
            center_idx = traj.shape[2] // 2

            # We use the scaled values directly for correlation analysis
            batch_cn0 = traj[:, idx_cn0, center_idx].cpu().numpy()
            batch_unc = traj[:, idx_unc, center_idx].cpu().numpy()
            batch_sv = ctx[:, idx_sv].cpu().numpy()

            feature_data["mean_cn0"].extend(batch_cn0)
            feature_data["mean_uncertainty"].extend(batch_unc)
            feature_data["sv_count"].extend(batch_sv)

    if not all_preds:
        print("No predictions generated during validation.")
        return float("inf")

    pred_residuals_m = np.concatenate(all_preds, axis=0)

    # ---------------------------------------------------------
    # Reconstruct DataFrame for Metric Calculation
    # ---------------------------------------------------------
    # We need to align the predictions with the original dataframe.
    # GNSSSequenceDataset filters out the start/end of trips based on window size.
    # We must replicate this filtering to get the corresponding ground truth.

    half_window = WINDOW_SIZE // 2
    grouped = val_df.groupby("tripId")
    trip_ids = val_df["tripId"].unique()

    val_indices = []
    for tid in trip_ids:
        grp = grouped.get_group(tid)
        n = len(grp)
        # The dataset iterates: range(half_window, n - half_window)
        start = half_window
        end = n - half_window
        if start < end:
            # Get global indices for this valid range
            val_indices.extend(grp.index[start:end].values)

    val_df_subset = val_df.loc[val_indices].copy()

    # Safety check
    if len(val_df_subset) != len(pred_residuals_m):
        print(
            f"Error: Alignment mismatch. DF: {len(val_df_subset)}, Preds: {len(pred_residuals_m)}"
        )
        return float("inf")

    # Apply predictions
    wls_lat = val_df_subset["wls_lat"].values
    wls_lon = val_df_subset["wls_lon"].values

    pred_res_lat_m = pred_residuals_m[:, 0]
    pred_res_lon_m = pred_residuals_m[:, 1]

    pred_lat_deg, pred_lon_deg = meters_to_degrees(
        pred_res_lat_m, pred_res_lon_m, wls_lat
    )

    val_df_subset["pred_lat"] = wls_lat + pred_lat_deg
    val_df_subset["pred_lon"] = wls_lon + pred_lon_deg

    # ---------------------------------------------------------
    # Calculate Metric
    # ---------------------------------------------------------
    metric = calculate_metric(val_df_subset)
    print(f"Final Validation Metric: {metric}")

    # ---------------------------------------------------------
    # Failure Analysis
    # ---------------------------------------------------------
    # Create analysis dataframe
    analysis_df = pd.DataFrame(
        {
            "error_m": val_df_subset["error_m"].values,
            "mean_cn0": feature_data["mean_cn0"],
            "mean_uncertainty": feature_data["mean_uncertainty"],
            "sv_count": feature_data["sv_count"],
        }
    )

    print("\nFailure Analysis (Spearman Correlation with Error):")
    corr = analysis_df.corr(method="spearman")["error_m"].drop("error_m")
    print(corr)

    return metric


def main():
    print("==================================================")
    print("STARTING PIPELINE EXECUTION")
    print("==================================================")

    # 1. Train Model
    # load_cached_data=True allows skipping raw data processing if parquet exists
    print("\n[Step 1] Training Model...")
    model, scaler = train_model(load_cached_data=True)

    # 2. Validation & Analysis
    print("\n[Step 2] Loading Validation Data...")
    val_df = get_dataset("val", load_cached_data=True)

    print("\n[Step 3] Validating...")
    metric = run_validation_and_analysis(model, scaler, val_df)

    # 3. Conditional Submission
    submission_threshold = 4.256982128481356
    print(f"\n[Step 4] Checking Threshold: {metric} vs {submission_threshold}")

    if metric < submission_threshold:
        print("Metric check passed. Generating submission...")
        generate_predictions(load_cached_data=True)
    else:
        print("Metric check failed. Skipping submission generation.")

    print("\nPipeline execution complete.")


if __name__ == "__main__":
    main()
