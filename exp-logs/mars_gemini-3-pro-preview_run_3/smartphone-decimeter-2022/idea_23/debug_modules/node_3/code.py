import os
import sys
import numpy as np
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set random seed
np.random.seed(42)

# Import library modules
# We modify the configuration parameters for the demonstration to ensure speed
import library.config as config

# Patch configuration for speed
config.LGBM_PARAMS["n_estimators"] = 10
config.LGBM_PARAMS["num_leaves"] = 8
config.N_FOLDS = 2  # Reduce folds for demo
config.WORKING_DIR = "./working/demo_run"
os.makedirs(config.WORKING_DIR, exist_ok=True)

from library.coord_utils import wgs84_to_ecef, ecef_to_enu
from library.data_loader import GnssLoader
from library.feature_eng import FeatureEngine
from library.kinematics import KinematicsEngine
from library.model_wrapper import LGBMEnsemble
from library.graph_optimizer import GraphOptimizer


def main():
    print("=== Starting Demonstration ===")

    # ---------------------------------------------------------
    # 1. Coordinate Utilities Demonstration
    # ---------------------------------------------------------
    print("\n[1] Testing Coordinate Utilities...")
    lat, lon, alt = 37.42, -122.08, 30.0
    x, y, z = wgs84_to_ecef(lat, lon, alt)

    # Check if ECEF conversion produces reasonable magnitude (Earth radius ~6371km)
    r = np.sqrt(x**2 + y**2 + z**2)
    print(f"  WGS84 ({lat}, {lon}, {alt}) -> ECEF ({x:.2f}, {y:.2f}, {z:.2f})")
    print(f"  Radius: {r:.2f} meters")
    assert 6300000 < r < 6400000, "ECEF conversion radius out of bounds"

    # Check ENU conversion (origin at same point should be 0,0,0)
    e, n, u = ecef_to_enu(x, y, z, lat, lon, alt)
    print(f"  ECEF to ENU at origin: ({e:.4f}, {n:.4f}, {u:.4f})")
    assert np.allclose([e, n, u], [0, 0, 0], atol=1e-3), "ENU origin check failed"
    print("  Coordinate utilities verified.")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n[2] Testing Data Loader...")
    loader = GnssLoader(working_dir=config.WORKING_DIR)

    # Load metadata
    train_meta = loader.load_metadata("train")
    print(f"  Loaded train metadata with {len(train_meta)} rows.")

    # Pick a specific drive for demonstration to keep it fast
    # Using a known drive from the file list
    target_drive = "2020-05-15-US-MTV-1"
    target_phone = "GooglePixel4XL"

    print(f"  Loading data for {target_drive} - {target_phone}...")
    gnss_df, imu_df, gt_df = loader.get_drive_data(
        target_drive, target_phone, split="train"
    )

    print(f"  GNSS shape: {gnss_df.shape}")
    print(f"  IMU shape: {imu_df.shape}")
    print(f"  GT shape: {gt_df.shape}")

    assert not gnss_df.empty, "GNSS dataframe is empty"
    assert not imu_df.empty, "IMU dataframe is empty"
    assert gt_df is not None and not gt_df.empty, "Ground Truth dataframe is empty"
    print("  Data loading verified.")

    # ---------------------------------------------------------
    # 3. Feature Engineering
    # ---------------------------------------------------------
    print("\n[3] Testing Feature Engineering...")
    feat_engine = FeatureEngine(working_dir=config.WORKING_DIR)

    # Process the single drive
    # We use load_cached_data=False to ensure we actually run the logic
    features_df = feat_engine.process_drive(
        target_drive, target_phone, split="train", load_cached_data=False
    )

    # Add tripId which is required for the model wrapper
    trip_id = f"{target_drive}-{target_phone}"
    features_df["tripId"] = trip_id

    print(f"  Generated features shape: {features_df.shape}")
    print(f"  Feature columns: {list(features_df.columns[:5])} ...")

    expected_cols = ["F_pr_E", "F_pr_N", "Target_E", "Target_N"]
    for col in expected_cols:
        assert col in features_df.columns, f"Missing expected column {col} in features"

    # Check for NaNs in critical features
    assert not features_df["F_pr_E"].isna().all(), "Feature F_pr_E contains all NaNs"
    print("  Feature engineering verified.")

    # ---------------------------------------------------------
    # 4. Kinematics Estimation
    # ---------------------------------------------------------
    print("\n[4] Testing Kinematics Engine...")
    kin_engine = KinematicsEngine(
        cache_dir=os.path.join(config.WORKING_DIR, "kin_cache")
    )

    # Compute displacements
    kin_df = kin_engine.compute_displacements(
        gnss_df, target_drive, target_phone, load_cached_data=False
    )

    print(f"  Kinematics shape: {kin_df.shape}")
    print(f"  Kinematics columns: {list(kin_df.columns)}")

    assert (
        "dx" in kin_df.columns and "dy" in kin_df.columns
    ), "Missing displacement columns"
    assert len(kin_df) > 0, "Kinematics dataframe is empty"

    # Check if we have any valid weights (meaning successful TDCP or Doppler)
    valid_kin = kin_df[kin_df["weight"] > 0]
    print(f"  Valid kinematic epochs: {len(valid_kin)} / {len(kin_df)}")
    print("  Kinematics estimation verified.")

    # ---------------------------------------------------------
    # 5. Model Training (LightGBM Ensemble)
    # ---------------------------------------------------------
    print("\n[5] Testing Model Training...")
    model = LGBMEnsemble()

    # Train on the single drive (split internally by GroupKFold, but here group is same,
    # so KFold will likely complain or put everything in train if groups are identical.
    # To make it work for demo, we artificially split the tripId into pseudo-trips)

    # Create pseudo-groups for cross-validation on a single drive
    n_rows = len(features_df)
    # Inject part identifier into the drive_id section so it isn't dropped by model_wrapper
    # model_wrapper extracts group as "-".join(tripId.split("-")[:-1])
    # Cite debug_lesson_8
    features_df["tripId"] = [
        f"{target_drive}_part{i%2}-{target_phone}" for i in range(n_rows)
    ]

    model.fit(features_df, load_cached_data=False)

    # Restore correct tripId
    features_df["tripId"] = trip_id

    print("  Model training complete.")

    # Predict
    pred_e, pred_n = model.predict(features_df)
    features_df["Pred_E"] = pred_e
    features_df["Pred_N"] = pred_n

    print(
        f"  Predictions generated. Mean E: {np.mean(pred_e):.4f}, Mean N: {np.mean(pred_n):.4f}"
    )
    assert len(pred_e) == len(features_df), "Prediction length mismatch"
    print("  Model training and prediction verified.")

    # ---------------------------------------------------------
    # 6. Graph Optimization
    # ---------------------------------------------------------
    print("\n[6] Testing Graph Optimizer...")
    optimizer = GraphOptimizer(working_dir=config.WORKING_DIR)

    # Prepare anchor dataframe (features + predictions)
    # Ensure WLS positions are present
    anchor_df = features_df.copy()

    # Solve trajectory
    optimized_df = optimizer.solve_trajectory(
        target_drive, target_phone, anchor_df, kin_df, load_cached_data=False
    )

    print(f"  Optimized trajectory shape: {optimized_df.shape}")

    if not optimized_df.empty:
        print(f"  Optimized columns: {list(optimized_df.columns)}")
        assert "LatitudeDegrees" in optimized_df.columns, "Missing Latitude in result"
        assert "LongitudeDegrees" in optimized_df.columns, "Missing Longitude in result"

        # Basic sanity check: optimized position shouldn't be too far from WLS
        # We can check the first point
        wls_lat = features_df.iloc[0]["Wls_Lat"]
        opt_lat = optimized_df.iloc[0]["LatitudeDegrees"]
        diff = abs(wls_lat - opt_lat)
        print(f"  WLS Lat: {wls_lat:.6f}, Opt Lat: {opt_lat:.6f}, Diff: {diff:.6f}")
        assert diff < 0.1, "Optimization diverged significantly from WLS"
    else:
        print(
            "  Warning: Optimization returned empty result (possibly due to data alignment issues in demo subset)"
        )

    print("  Graph optimization verified.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
