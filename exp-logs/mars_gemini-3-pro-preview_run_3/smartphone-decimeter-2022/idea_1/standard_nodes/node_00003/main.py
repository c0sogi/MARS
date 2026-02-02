import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import (
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    MODEL_SAVE_PATH,
    INPUT_DIM,
    WORK_DIR,
    SATELLITE_FEATURES,
    TOP_K_SATELLITES,
)
from library.train import run_training
from library.data import load_dataset
from library.model import ResidualMLP
from library.utils import ecef_to_geodetic
from library.inference import predict_and_submit


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r


import shutil


def main():
    # Clear cache to ensure data cleaning is applied (Cite debug_lesson_1)
    if os.path.exists(WORK_DIR):
        print(f"Cleaning cache directory: {WORK_DIR}")
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Train the model
    # -------------------------------------------------------------------------
    print("Step 1: Training Model...")
    # run_training handles data loading, caching, and the training loop.
    # It saves the best model to MODEL_SAVE_PATH and returns the model instance.
    model = run_training(load_cached_data=True)

    # -------------------------------------------------------------------------
    # 2. Validation Assessment
    # -------------------------------------------------------------------------
    print("\nStep 2: Validation Assessment...")

    # Load validation data
    # load_cached_data=True attempts to load parquet files if they exist
    val_dataset = load_dataset(mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_residuals = []

    # Inference on validation set
    with torch.no_grad():
        for batch in val_loader:
            # val_dataset returns (features, targets)
            features, _ = batch
            features = features.to(device)
            outputs = model(features)
            all_residuals.append(outputs.cpu().numpy())

    residuals = np.concatenate(all_residuals, axis=0)

    # Reconstruction
    meta_df = val_dataset.meta.copy()

    # Extract baseline WLS ECEF coordinates
    wls_x = meta_df["WlsPositionXEcefMeters"].values
    wls_y = meta_df["WlsPositionYEcefMeters"].values
    wls_z = meta_df["WlsPositionZEcefMeters"].values

    # Add predicted residuals to baseline
    pred_x = wls_x + residuals[:, 0]
    pred_y = wls_y + residuals[:, 1]
    pred_z = wls_z + residuals[:, 2]

    # Convert ECEF to Geodetic coordinates
    pred_lat, pred_lon, _ = ecef_to_geodetic(pred_x, pred_y, pred_z)

    # Calculate Haversine Distance Errors
    gt_lat = meta_df["LatitudeDegrees"].values
    gt_lon = meta_df["LongitudeDegrees"].values

    errors = haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)

    # Add errors to metadata for grouping
    meta_df["error"] = errors

    # Calculate Metric: Mean of (Mean(50th, 95th) per phone)
    # Group by tripId (which corresponds to a unique phone-drive)
    score_per_trip = []
    for trip_id, group in meta_df.groupby("tripId"):
        errs = group["error"].values
        p50 = np.percentile(errs, 50)
        p95 = np.percentile(errs, 95)
        avg_score = (p50 + p95) / 2
        score_per_trip.append(avg_score)

    final_metric = np.mean(score_per_trip)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 3. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nStep 3: Failure Analysis...")

    # Get feature matrix from validation dataset (convert tensor to numpy)
    features_np = val_dataset.features.numpy()

    # Reconstruct feature names based on data.py logic
    # Order: Feature1_Sat0, Feature1_Sat1... Feature2_Sat0...
    feature_names = []
    for feat in SATELLITE_FEATURES:
        for k in range(TOP_K_SATELLITES):
            feature_names.append(f"{feat}_{k}")

    n_samples = features_np.shape[0]
    if n_samples > 0:
        # Calculate correlation between each feature and the error magnitude
        # Center data
        f_centered = features_np - features_np.mean(axis=0)
        e_centered = errors - errors.mean()

        # Compute covariance
        covariance = np.dot(f_centered.T, e_centered) / (n_samples - 1)

        # Compute standard deviations
        f_std = features_np.std(axis=0)
        e_std = errors.std()

        # Avoid division by zero
        f_std[f_std == 0] = 1e-9
        if e_std == 0:
            e_std = 1e-9

        correlations = covariance / (f_std * e_std)

        # Create DataFrame for display
        corr_df = pd.DataFrame({"Feature": feature_names, "Correlation": correlations})

        # Sort by absolute correlation
        corr_df["AbsCorr"] = corr_df["Correlation"].abs()
        corr_df = corr_df.sort_values("AbsCorr", ascending=False)

        print("Top 10 Features correlated with Error Magnitude:")
        print(corr_df[["Feature", "Correlation"]].head(10).to_string(index=False))
    else:
        print("No validation samples available for analysis.")

    # -------------------------------------------------------------------------
    # 4. Submission
    # -------------------------------------------------------------------------
    print("\nStep 4: Generating Submission...")
    # predict_and_submit handles loading test data, inference, and saving CSV
    predict_and_submit(load_cached_data=True)


if __name__ == "__main__":
    main()
