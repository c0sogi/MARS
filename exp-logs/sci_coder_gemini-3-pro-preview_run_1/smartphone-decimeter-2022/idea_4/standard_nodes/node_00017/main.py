import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, enu_to_geodetic
from library.data_processing import process_dataset
from library.dataset import GNSSSequenceDataset, collate_padded
from library.trainer import train_model
from library.inference import generate_submission


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine distance between two sets of coordinates.
    """
    R = 6371000.0  # Radius of Earth in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def calculate_metric(df):
    """
    Computes the competition metric: Mean of (50th + 95th percentile) errors averaged across phones.
    """
    # Calculate distance error in meters
    df["error_m"] = haversine_distance(
        df["LatitudeDegrees"].values,
        df["LongitudeDegrees"].values,
        df["Pred_Lat"].values,
        df["Pred_Lon"].values,
    )

    # Group by trip (drive_id + phone_name)
    trips = df.groupby(["drive_id", "phone_name"])

    trip_scores = []
    for _, group in trips:
        errors = group["error_m"].values
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        trip_score = (p50 + p95) / 2
        trip_scores.append(trip_score)

    final_metric = np.mean(trip_scores)
    return final_metric, df


def run():
    # 1. Configure for Fast Baseline
    # Limit epochs to ensure completion within time limits
    Config.EPOCHS = 5
    # Adjust batch size for speed/memory balance
    Config.BATCH_SIZE = 32

    print(
        f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}, Device={Config.DEVICE}"
    )

    # 2. Train Model
    print("\n--- Starting Training ---")
    # load_cached_data=True uses the pre-processed parquet/npz files in ./working
    model, feature_stats = train_model(load_cached_data=True)

    # 3. Validation Inference
    print("\n--- Starting Validation Inference ---")
    # Load validation dataframe (features + ground truth)
    val_df = process_dataset(
        Config.VAL_METADATA_PATH, load_cached_data=True, split_name="val"
    )

    # Create dataset using training statistics for normalization
    val_dataset = GNSSSequenceDataset(
        val_df, split_name="val", feature_stats=feature_stats, load_cached_data=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_padded,
    )

    model.eval()
    device = Config.DEVICE
    results = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            lengths = batch["lengths"]
            timestamps = batch["timestamps"]
            drive_ids = batch["drive_id"]
            phone_names = batch["phone_name"]

            # Forward pass: Predict ENU residuals (North, East)
            outputs = model(features).cpu().numpy()  # Shape: (B, C, L)

            for i in range(len(drive_ids)):
                length = lengths[i]
                drive_id = drive_ids[i]
                phone_name = phone_names[i]

                # Extract valid sequence (remove padding)
                # outputs[i] is (C, L_padded) -> Transpose to (L_padded, C)
                pred_seq = outputs[i].T[:length]
                time_seq = timestamps[i][:length].numpy()

                for t, (lat_res, lon_res) in enumerate(pred_seq):
                    results.append(
                        {
                            "drive_id": drive_id,
                            "phone_name": phone_name,
                            "UnixTimeMillis": time_seq[t],
                            "lat_res_pred": lat_res,
                            "lon_res_pred": lon_res,
                        }
                    )

    pred_val_df = pd.DataFrame(results)

    # Merge predictions with Ground Truth and Baseline in val_df
    # Ensure join keys are integers
    val_df["UnixTimeMillis"] = val_df["UnixTimeMillis"].astype(np.int64)
    pred_val_df["UnixTimeMillis"] = pred_val_df["UnixTimeMillis"].astype(np.int64)

    merged_val = pd.merge(
        val_df,
        pred_val_df,
        on=["drive_id", "phone_name", "UnixTimeMillis"],
        how="inner",
    )

    # Reconstruct Predicted Geodetic Coordinates from Baseline + Predicted Residuals
    pred_lats, pred_lons = enu_to_geodetic(
        merged_val["lat_res_pred"].values,
        merged_val["lon_res_pred"].values,
        merged_val["BaselineLat"].values,
        merged_val["BaselineLon"].values,
    )

    merged_val["Pred_Lat"] = pred_lats
    merged_val["Pred_Lon"] = pred_lons

    # 4. Calculate Metric
    metric, analyzed_df = calculate_metric(merged_val)
    print(f"Final Validation Metric: {metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Identify feature columns (excluding metadata and targets)
    exclude_cols = [
        "drive_id",
        "phone_name",
        "UnixTimeMillis",
        "LatitudeDegrees",
        "LongitudeDegrees",
        "gnss_path",
        "imu_path",
        "UnixTimeMillis_rounded",
        "BaselineLat",
        "BaselineLon",
        "BaselineAlt",
        "lat_res_m",
        "lon_res_m",
        "Pred_Lat",
        "Pred_Lon",
        "error_m",
        "lat_res_pred",
        "lon_res_pred",
    ]
    feature_cols = [c for c in analyzed_df.columns if c not in exclude_cols]
    # Filter for numeric columns only
    feature_cols = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(analyzed_df[c])
    ]

    correlations = {}
    for col in feature_cols:
        if analyzed_df[col].std() > 0:  # Skip constant columns
            corr = analyzed_df[col].corr(analyzed_df["error_m"])
            correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Correlations with Error Magnitude:")
    for name, val in sorted_corr[:5]:
        print(f"  {name}: {val:.4f}")

    # 6. Submission
    THRESHOLD = 3.8442371867640412
    if metric < THRESHOLD:
        print(
            f"\nMetric ({metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, feature_stats, load_cached_data=True)
    else:
        print(
            f"\nMetric ({metric}) is NOT below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
