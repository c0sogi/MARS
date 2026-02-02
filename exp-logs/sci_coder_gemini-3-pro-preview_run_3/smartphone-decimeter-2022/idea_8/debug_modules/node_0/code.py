import sys
import os
import numpy as np
import pandas as pd
import warnings

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.data_loader import get_processed_dataset
from library.model_lgbm import train_residual_model, predict_residuals
from library.kalman_filter import RobustKalmanSmoother
from library.evaluation import compute_competition_metric
from library.coord_utils import wgs84_to_ecef, ecef_to_wgs84, geodetic_to_enu

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    print("Starting demonstration script...")
    set_seed(Config.SEED)

    # 1. Configuration Overrides for Speed
    print("\n--- 1. Configuring for Speed ---")
    # Enable DEBUG mode to load only a small subset of drives
    Config.DEBUG = True
    # Reduce estimators to ensure training completes very quickly
    Config.N_ESTIMATORS = 10
    Config.LGBM_PARAMS["n_estimators"] = 10

    # Use a specific subdirectory in working for this demo to avoid conflicts
    demo_dir = os.path.join(Config.WORKING_DIR, "demo_run")
    os.makedirs(demo_dir, exist_ok=True)

    # Override paths to use demo directory for caching
    Config.TRAIN_FEATURES_PATH = os.path.join(demo_dir, "train_features.parquet")

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Estimators: {Config.N_ESTIMATORS}")

    # 2. Data Loading & Feature Engineering
    print("\n--- 2. Loading and Processing Data ---")
    # This function handles:
    # - Loading metadata
    # - Reading raw GNSS/IMU files
    # - Feature extraction (via feature_eng.py logic embedded in data_loader)
    # - Merging and Target computation
    df_train = get_processed_dataset("train", load_cached_data=False)

    print(f"Loaded training data shape: {df_train.shape}")

    # Basic Validations
    assert not df_train.empty, "Training dataframe is empty!"
    # Check for some expected features and targets
    expected_cols = [
        "target_E",
        "target_N",
        "wls_lat",
        "wls_lon",
        "cn0_mean",
        "sv_count",
    ]
    for col in expected_cols:
        assert col in df_train.columns, f"Missing expected column: {col}"

    print("Data loaded and features verified.")

    # 3. Model Training (LightGBM)
    print("\n--- 3. Training Residual Models ---")

    # Define features to use (exclude metadata and targets)
    feature_cols = [
        c
        for c in df_train.columns
        if c
        not in [
            "tripId",
            "UnixTimeMillis",
            "drive_id",
            "phone_name",
            "gnss_path",
            "imu_path",
            "gt_path",
            "LatitudeDegrees",
            "LongitudeDegrees",
            "wls_lat",
            "wls_lon",
            "wls_alt",
            "target_E",
            "target_N",
        ]
    ]
    # Filter out any remaining object columns
    feature_cols = [c for c in feature_cols if df_train[c].dtype != "object"]

    print(f"Training with {len(feature_cols)} features.")

    # Train East Model
    print("Training East Model...")
    models_E, oof_E, score_E = train_residual_model(
        df_train, feature_cols, "target_E", n_splits=2  # Reduced splits for speed
    )

    # Train North Model
    print("Training North Model...")
    models_N, oof_N, score_N = train_residual_model(
        df_train, feature_cols, "target_N", n_splits=2
    )

    print(f"East MAE: {score_E:.4f}")
    print(f"North MAE: {score_N:.4f}")

    # 4. Prediction & Coordinate Reconstruction
    print("\n--- 4. Generating Predictions ---")

    # We will use a subset of the training data to simulate a test prediction phase
    subset = df_train.head(500).copy()

    # Predict residuals
    pred_E = predict_residuals(subset, feature_cols, models_E)
    pred_N = predict_residuals(subset, feature_cols, models_N)

    # Reconstruct predictions: Lat/Lon = WLS + Predicted Residuals
    # We approximate the conversion from meters to degrees
    wls_lat = subset["wls_lat"].values
    wls_lon = subset["wls_lon"].values

    R = 6378137.0  # Earth Radius
    dLat = np.degrees(pred_N / R)
    dLon = np.degrees(pred_E / (R * np.cos(np.radians(wls_lat))))

    subset["pred_lat"] = wls_lat + dLat
    subset["pred_lon"] = wls_lon + dLon

    # Verify predictions are valid
    assert not subset["pred_lat"].isna().any()
    assert not subset["pred_lon"].isna().any()
    print("Predictions generated.")

    # 5. Kalman Smoothing
    print("\n--- 5. Applying Kalman Smoothing ---")

    smoother = RobustKalmanSmoother()

    # Select one trip for smoothing demonstration
    trip_id = subset["tripId"].iloc[0]
    trip_df = subset[subset["tripId"] == trip_id].copy()

    # Rename columns to match what smoother expects
    trip_df = trip_df.rename(columns={"pred_lat": "lat", "pred_lon": "lon"})

    print(f"Smoothing trip: {trip_id} with {len(trip_df)} points")

    smoothed_df = smoother.apply(trip_df)

    # Check if smoothing actually modified the path
    diff = np.abs(smoothed_df["lat"] - trip_df["lat"]).sum()
    print(f"Total Latitude adjustment by smoother: {diff:.6f}")

    # 6. Evaluation
    print("\n--- 6. Evaluation ---")

    # Prepare DataFrames for metric computation
    # Predictions DataFrame
    df_pred = smoothed_df[["tripId", "UnixTimeMillis", "lat", "lon"]].rename(
        columns={"lat": "LatitudeDegrees", "lon": "LongitudeDegrees"}
    )

    # Ground Truth DataFrame (from the original subset)
    df_gt = subset[subset["tripId"] == trip_id][
        ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    ]

    score = compute_competition_metric(df_pred, df_gt)
    print(f"Competition Metric Score for trip {trip_id}: {score:.4f}")

    assert not np.isnan(score), "Score should be a valid number"

    # 7. Coordinate Utils Verification
    print("\n--- 7. Verifying Coordinate Utilities ---")
    lat, lon, alt = 37.42, -122.08, 30.0
    x, y, z = wgs84_to_ecef(lat, lon, alt)
    lat_new, lon_new, alt_new = ecef_to_wgs84(x, y, z)

    print(f"Original: {lat}, {lon}, {alt}")
    print(f"Converted: {lat_new:.6f}, {lon_new:.6f}, {alt_new:.6f}")

    assert np.isclose(lat, lat_new, atol=1e-5)
    assert np.isclose(lon, lon_new, atol=1e-5)
    assert np.isclose(alt, alt_new, atol=1e-3)
    print("Coordinate conversion verified.")

    print("\nDemonstration complete.")


if __name__ == "__main__":
    main()
