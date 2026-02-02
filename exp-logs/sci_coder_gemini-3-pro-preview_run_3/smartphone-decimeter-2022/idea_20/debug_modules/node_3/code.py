import os
import pandas as pd
import numpy as np
import shutil
import warnings
from library.utils import CoordinateTransformer, MetricCalculator, IOHelper
from library.data_loader import GnssDataset
from library.kinematics import KinematicsEngine
from library.features import FeatureEngineer
from library.model import ResidualRegressor
from library.optimizer import TrajectoryOptimizer

# Configuration
CACHE_DIR = "./working/idea_20/"
METADATA_PATH = "./metadata/train_metadata.csv"
INPUT_DIR = "./input"

# Suppress warnings
warnings.filterwarnings("ignore")


def clean_cache():
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)


def run_demo():
    print("--- Starting Library Demonstration ---")

    # 1. Test Utils
    print("\n[1] Testing Utils...")
    # Test Coordinate Transformer
    lat, lon, alt = 37.42, -122.08, 30.0
    x, y, z = CoordinateTransformer.wgs84_to_ecef(lat, lon, alt)
    lat_r, lon_r, alt_r = CoordinateTransformer.ecef_to_wgs84(x, y, z)

    assert np.isclose(lat, lat_r), "Lat conversion failed"
    assert np.isclose(lon, lon_r), "Lon conversion failed"
    assert np.isclose(alt, alt_r), "Alt conversion failed"
    print("CoordinateTransformer: OK")

    # Test Metric Calculator
    dist = MetricCalculator.haversine_distance(0, 0, 0, 1)
    assert dist > 0, "Haversine distance failed"
    print("MetricCalculator: OK")

    # 2. Data Loading (Single Trip)
    print("\n[2] Testing Data Loader...")
    if not os.path.exists(METADATA_PATH):
        print("Metadata not found. Skipping data dependent tests.")
        return

    df_meta = pd.read_csv(METADATA_PATH)
    # Pick the first trip
    sample_trip = df_meta.iloc[0]
    trip_id = sample_trip["tripId"]
    print(f"Selected Trip: {trip_id}")

    loader = GnssDataset(mode="train", root_dir=INPUT_DIR)

    # Load raw GNSS/IMU for this trip directly
    df_raw = loader._process_trip(
        trip_id,
        sample_trip["gnss_path"],
        sample_trip["imu_path"],
        sample_trip["gt_path"],
    )

    # Manually add context identifiers for the demo
    df_raw["drive_id"] = sample_trip["drive_id"]
    df_raw["phone_name"] = sample_trip["phone_name"]

    assert not df_raw.empty, "Loaded raw data is empty"
    assert "tripId" in df_raw.columns, "tripId column missing"
    print(f"Loaded {len(df_raw)} rows of raw data.")

    # 3. Kinematics
    print("\n[3] Testing Kinematics Engine...")
    kine = KinematicsEngine()
    # Process trip to get velocities and tdcp
    df_kinematics = kine.process_trip(df_raw, load_cached_data=False)

    assert not df_kinematics.empty, "Kinematics output is empty"
    expected_k_cols = ["v_east_dop", "d_east_tdcp", "valid_dop"]
    for c in expected_k_cols:
        assert c in df_kinematics.columns, f"Missing kinematics column: {c}"
    print(f"Kinematics processed. Shape: {df_kinematics.shape}")

    # 4. Feature Engineering
    print("\n[4] Testing Feature Engineer...")
    fe = FeatureEngineer()
    df_features = fe.process_trip(df_raw, load_cached_data=False)

    assert not df_features.empty, "Feature output is empty"
    # Check for a specific feature
    assert "CN0_mean" in df_features.columns, "CN0_mean feature missing"
    print(f"Features generated. Shape: {df_features.shape}")

    # 5. Model Training
    print("\n[5] Testing Model Training...")
    # Prepare a small training set for the regressor
    # We need to merge features with Ground Truth targets

    # Get GT from raw load (loader._process_trip merges GT for train mode)
    # We need to aggregate raw GT to epoch level (features are epoch level)
    # The raw df already has GT columns repeated for satellites, so we drop duplicates by time
    df_epoch_gt = df_raw.drop_duplicates(subset=["utcTimeMillis"])

    # Merge features with GT info
    df_train_subset = pd.merge(
        df_epoch_gt, df_features, on=["tripId", "utcTimeMillis"], how="inner"
    )

    # Initialize Regressor
    regressor = ResidualRegressor(n_estimators=2, num_leaves=5)  # Minimal for speed

    # Compute targets (ENU residuals)
    df_train_subset = regressor._compute_targets(df_train_subset)

    # Artificial group split for demo CV to satisfy GroupKFold n_splits=2
    mid_point = len(df_train_subset) // 2
    df_train_subset.loc[:mid_point, "drive_id"] = "drive_demo_1"
    df_train_subset.loc[mid_point:, "drive_id"] = "drive_demo_2"

    assert "target_east" in df_train_subset.columns, "Target computation failed"

    # Save this subset to cache so train_cross_validation picks it up
    # The model class looks for 'prepared_train_data.parquet'
    IOHelper.save_parquet(df_train_subset, "prepared_train_data.parquet")

    # Run training
    regressor.train_cross_validation(n_splits=2, load_cached_data=True)
    assert len(regressor.models_east) == 2, "Model training failed to produce models"
    print("Model training simulation complete.")

    # 6. Optimization
    print("\n[6] Testing Trajectory Optimizer...")
    optimizer = TrajectoryOptimizer()

    # Create a dummy prediction dataframe (using WLS as 'prediction')
    # We need 'LatitudeDegrees' and 'LongitudeDegrees' which are in df_train_subset (GT)
    # Let's use WLS position converted to Lat/Lon as the "prediction" to optimize

    wls_x = df_train_subset["WlsPositionXEcefMeters"].values
    wls_y = df_train_subset["WlsPositionYEcefMeters"].values
    wls_z = df_train_subset["WlsPositionZEcefMeters"].values

    lat_wls, lon_wls, _ = CoordinateTransformer.ecef_to_wgs84(wls_x, wls_y, wls_z)

    df_pred = pd.DataFrame(
        {
            "tripId": df_train_subset["tripId"],
            "UnixTimeMillis": df_train_subset["utcTimeMillis"],
            "LatitudeDegrees": lat_wls,
            "LongitudeDegrees": lon_wls,
        }
    )

    # Run optimization on the single trip
    # We pass the kinematics dataframe we computed earlier
    # Note: df_kinematics uses 'utcTimeMillis', df_pred uses 'UnixTimeMillis'.
    # They should match in this dataset.

    df_optimized = optimizer._optimize_trip(trip_id, df_pred, df_kinematics)

    assert len(df_optimized) == len(df_pred), "Optimization output length mismatch"
    assert (
        "LatitudeDegrees" in df_optimized.columns
    ), "Optimization output missing columns"

    # Check if values changed (optimization actually did something)
    diff = np.abs(df_optimized["LatitudeDegrees"] - df_pred["LatitudeDegrees"]).sum()
    print(f"Total Latitude Adjustment: {diff:.6f}")

    print("Optimization complete.")
    print("\nAll demonstrations passed successfully.")


if __name__ == "__main__":
    # Ensure clean state
    clean_cache()
    run_demo()
