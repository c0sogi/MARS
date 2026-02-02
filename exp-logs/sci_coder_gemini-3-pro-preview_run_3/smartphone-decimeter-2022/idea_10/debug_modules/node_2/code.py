import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
from unittest.mock import patch

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 1. Monkey Patching for Speed and Silence ---
# We need to modify configuration constants and silence tqdm without changing library files.

import library.config as config

# Reduce optimization epochs for demonstration speed
config.OPTIMIZER_EPOCHS = 10
# Reduce LightGBM estimators for speed
config.LGBM_PARAMS["n_estimators"] = 10
config.LGBM_PARAMS["verbose"] = -1


# Silence tqdm to meet output requirements
def silent_tqdm(iterable, *args, **kwargs):
    return iterable


import tqdm

tqdm.tqdm = silent_tqdm

# --- Import Library Modules ---
from library.data_loader import (
    load_metadata,
    load_gnss_raw,
    load_imu_raw,
    load_ground_truth,
)
from library.feature_engineering import (
    estimate_doppler_velocity,
    create_pointwise_features,
    prepare_dataset,  # We will replicate logic manually for granular control over subsampling
)
from library.utils import (
    geodetic_to_ecef,
    ecef_to_geodetic,
    ecef_to_enu,
    enu_to_ecef,
    haversine_distance,
)
from library.model import ResidualRegressor
from library.optimizer import optimize_trajectory

# --- Main Execution ---

if __name__ == "__main__":
    print("--- Starting Library Demonstration ---")

    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # ---------------------------------------------------------
    # 1. Data Loading (Subsampled)
    # ---------------------------------------------------------
    print("\n[1] Loading Metadata and Raw Data...")

    # Load full train metadata
    full_meta = load_metadata("train")

    # Select a single drive for demonstration to ensure speed
    sample_drive_id = full_meta["drive_id"].unique()[0]
    print(f"Selected Sample Drive: {sample_drive_id}")

    # Filter metadata for this drive
    sample_meta = full_meta[full_meta["drive_id"] == sample_drive_id].copy()
    print(f"Metadata rows for sample: {len(sample_meta)}")

    # Load raw data for this subset
    # Note: We pass the filtered metadata to load only relevant files
    gnss_df = load_gnss_raw(
        sample_meta, split_name="demo_train", load_cached_data=False
    )
    imu_df = load_imu_raw(sample_meta, split_name="demo_train", load_cached_data=False)
    gt_df = load_ground_truth(
        sample_meta, split_name="demo_train", load_cached_data=False
    )

    print(f"Loaded GNSS rows: {len(gnss_df)}")
    print(f"Loaded IMU rows: {len(imu_df)}")
    print(f"Loaded GT rows: {len(gt_df)}")

    assert not gnss_df.empty, "GNSS data should not be empty"
    assert not gt_df.empty, "Ground Truth data should not be empty"

    # ---------------------------------------------------------
    # 2. Feature Engineering
    # ---------------------------------------------------------
    print("\n[2] Feature Engineering...")

    # A. Doppler Velocity Estimation
    print("Estimating Doppler Velocity...")
    doppler_df = estimate_doppler_velocity(gnss_df)

    # Validate Doppler output
    assert "v_doppler_x" in doppler_df.columns
    assert len(doppler_df) > 0
    print("Doppler estimation complete.")

    # B. Point-wise Features
    print("Creating Point-wise Features...")
    features_df = create_pointwise_features(gnss_df, imu_df)

    # Validate Feature output
    assert "gnss_Cn0DbHz_mean" in features_df.columns
    print("Feature creation complete.")

    # C. Merge Data
    print("Merging Data...")
    # Merge Features + Doppler
    full_df = pd.merge(
        features_df,
        doppler_df,
        on=["drive_id", "phone_name", "UnixTimeMillis"],
        how="left",
    )

    # Add WLS Baseline (First WLS per epoch)
    wls_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    wls_df = (
        gnss_df.groupby(["drive_id", "phone_name", "UnixTimeMillis"])[wls_cols]
        .first()
        .reset_index()
    )

    full_df = pd.merge(
        full_df, wls_df, on=["drive_id", "phone_name", "UnixTimeMillis"], how="left"
    )

    # Merge with Ground Truth to align targets
    # We use inner join on UnixTimeMillis to ensure we have labels
    train_df = pd.merge(
        gt_df[
            [
                "drive_id",
                "phone_name",
                "UnixTimeMillis",
                "LatitudeDegrees",
                "LongitudeDegrees",
                "AltitudeMeters",
                "tripId",
            ]
        ],
        full_df,
        on=["drive_id", "phone_name", "UnixTimeMillis"],
        how="inner",
    )

    # Fill missing WLS (if any) for robustness
    train_df[wls_cols] = (
        train_df[wls_cols].fillna(method="ffill").fillna(method="bfill")
    )
    train_df = train_df.dropna(subset=wls_cols)

    print(f"Final Training DataFrame Shape: {train_df.shape}")

    # ---------------------------------------------------------
    # 3. Target Calculation (ENU Residuals)
    # ---------------------------------------------------------
    print("\n[3] Calculating Targets (ENU Residuals)...")

    # Convert GT (Lat, Lon, Alt) to ECEF
    gt_lat = train_df["LatitudeDegrees"].values
    gt_lon = train_df["LongitudeDegrees"].values
    gt_alt = train_df["AltitudeMeters"].fillna(0).values

    gt_x, gt_y, gt_z = geodetic_to_ecef(gt_lat, gt_lon, gt_alt)

    # Calculate ECEF Residuals (GT - WLS)
    wls_x = train_df["WlsPositionXEcefMeters"].values
    wls_y = train_df["WlsPositionYEcefMeters"].values
    wls_z = train_df["WlsPositionZEcefMeters"].values

    res_x = gt_x - wls_x
    res_y = gt_y - wls_y
    res_z = gt_z - wls_z

    # Rotate residuals to ENU frame using GT position as reference
    # Note: We use the utility function ecef_to_enu logic manually or adapt it.
    # The utility `ecef_to_enu` converts absolute ECEF points to ENU relative to ref.
    # Here we have a vector `res` in ECEF. We want that vector in ENU.
    # This is equivalent to `ecef_to_enu(gt_x, gt_y, gt_z, ref=wls)`? No.
    # It is equivalent to `ecef_to_enu(gt_x, gt_y, gt_z, ref=gt)` which is 0,0,0.
    # We want the vector representation.
    # Let's use the logic implemented in `prepare_dataset` which manually rotates the difference vector.

    lat_rad = np.radians(gt_lat)
    lon_rad = np.radians(gt_lon)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    target_e = -sin_lon * res_x + cos_lon * res_y
    target_n = -sin_lat * cos_lon * res_x - sin_lat * sin_lon * res_y + cos_lat * res_z

    train_df["target_e"] = target_e
    train_df["target_n"] = target_n

    print(
        f"Target Mean East: {np.mean(target_e):.4f}, Mean North: {np.mean(target_n):.4f}"
    )

    # ---------------------------------------------------------
    # 4. Model Training & Prediction
    # ---------------------------------------------------------
    print("\n[4] Training Residual Model...")

    # Prepare Features
    exclude_cols = [
        "tripId",
        "drive_id",
        "phone_name",
        "UnixTimeMillis",
        "LatitudeDegrees",
        "LongitudeDegrees",
        "AltitudeMeters",
        "target_e",
        "target_n",
    ]
    feature_cols = [
        c
        for c in train_df.columns
        if c not in exclude_cols and train_df[c].dtype != "object"
    ]

    # Simple Train/Test split (first 80% train, last 20% test)
    split_idx = int(len(train_df) * 0.8)
    X_train = train_df.iloc[:split_idx][feature_cols]
    y_e_train = train_df.iloc[:split_idx]["target_e"]
    y_n_train = train_df.iloc[:split_idx]["target_n"]

    X_val = train_df.iloc[split_idx:][feature_cols]
    y_e_val = train_df.iloc[split_idx:]["target_e"]
    y_n_val = train_df.iloc[split_idx:]["target_n"]

    # Train
    model = ResidualRegressor(config.LGBM_PARAMS)
    model.fit(X_train, y_e_train, y_n_train, X_val, y_e_val, y_n_val)

    # Predict on Validation
    pred_e, pred_n = model.predict(X_val)

    mae_e = np.mean(np.abs(y_e_val - pred_e))
    mae_n = np.mean(np.abs(y_n_val - pred_n))
    print(f"Validation MAE: East={mae_e:.4f}, North={mae_n:.4f}")

    # Add predictions to dataframe for optimization step
    # We'll use the validation set for optimization demo
    val_df = train_df.iloc[split_idx:].copy()
    val_df["pred_e"] = pred_e
    val_df["pred_n"] = pred_n

    # ---------------------------------------------------------
    # 5. Global Trajectory Optimization
    # ---------------------------------------------------------
    print("\n[5] Running Global Trajectory Optimization...")

    # Optimize
    # Note: We monkey-patched OPTIMIZER_EPOCHS to 10 for speed
    optimized_df = optimize_trajectory(val_df)

    # Verify results
    assert "LatitudeDegrees" in optimized_df.columns
    assert "LongitudeDegrees" in optimized_df.columns
    assert len(optimized_df) == len(val_df)

    # Check if optimization actually moved points from WLS baseline
    # Convert WLS back to Geodetic to compare
    wls_lat, wls_lon, _ = ecef_to_geodetic(
        val_df["WlsPositionXEcefMeters"].values,
        val_df["WlsPositionYEcefMeters"].values,
        val_df["WlsPositionZEcefMeters"].values,
    )

    diff_lat = np.mean(np.abs(optimized_df["LatitudeDegrees"].values - wls_lat))
    diff_lon = np.mean(np.abs(optimized_df["LongitudeDegrees"].values - wls_lon))

    print(f"Mean shift from WLS Baseline -> Lat: {diff_lat:.6f}, Lon: {diff_lon:.6f}")
    assert diff_lat > 0 or diff_lon > 0, "Optimization did not adjust positions!"

    # ---------------------------------------------------------
    # 6. Utility Functions Verification
    # ---------------------------------------------------------
    print("\n[6] Verifying Utility Functions...")

    # Test Coordinate Transforms Round-trip
    lat, lon, alt = 37.42, -122.08, 30.0
    x, y, z = geodetic_to_ecef(lat, lon, alt)
    lat_out, lon_out, alt_out = ecef_to_geodetic(x, y, z)

    print(f"Original: {lat}, {lon}, {alt}")
    print(f"Roundtrip: {lat_out:.6f}, {lon_out:.6f}, {alt_out:.6f}")

    assert np.isclose(lat, lat_out, atol=1e-5)
    assert np.isclose(lon, lon_out, atol=1e-5)
    assert np.isclose(alt, alt_out, atol=1e-3)

    # Test Haversine
    dist = haversine_distance(0, 0, 0, 1)  # 1 degree longitude at equator ~ 111km
    print(f"Distance 1 deg lon at equator: {dist:.2f} meters")
    assert 111000 < dist < 112000

    # Test ENU conversion consistency
    # ENU of a point relative to itself should be 0,0,0
    e, n, u = ecef_to_enu(x, y, z, lat, lon, alt)
    assert np.allclose([e, n, u], [0, 0, 0], atol=1e-3)

    print("\n--- Demonstration Complete Successfully ---")
