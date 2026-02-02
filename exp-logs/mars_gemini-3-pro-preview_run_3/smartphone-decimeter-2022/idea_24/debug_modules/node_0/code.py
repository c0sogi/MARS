import sys
import os
import numpy as np
import pandas as pd
import shutil

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import llh_to_ecef, ecef_to_llh, ecef_to_enu, haversine_distance
from library.data_io import load_metadata, load_drive_data
from library.feature_engineering import compute_geometric_features
from library.kinematics import compute_tdcp_odometry
from library.model import train_models, generate_submission
from library.optimization import run_global_optimization


def run_demonstration():
    print("### Starting GNSS Localization Library Demonstration ###")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Testing
    # ---------------------------------------------------------
    print("\n[1] Configuring environment...")
    # Enable debug mode to sample a small subset of data
    Config.DEBUG = True
    # Process only 50 rows from metadata (very small subset for speed)
    Config.DEBUG_SAMPLE_SIZE = 50
    # Reduce LightGBM complexity for instant training
    Config.LGBM_PARAMS["n_estimators"] = 2
    Config.LGBM_PARAMS["min_child_samples"] = 1
    Config.LGBM_PARAMS["num_leaves"] = 4

    # Ensure working directories are clean or exist
    if os.path.exists(Config.WORKING_DIR):
        # We don't delete it to avoid permission issues, just ensure it exists
        pass
    else:
        os.makedirs(Config.WORKING_DIR)

    print("Configuration updated for rapid execution.")

    # ---------------------------------------------------------
    # 2. Utils Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test WGS84 <-> ECEF conversion
    lat_orig, lon_orig, alt_orig = 37.4275, -122.1697, 30.0
    x, y, z = llh_to_ecef(lat_orig, lon_orig, alt_orig)
    lat_new, lon_new, alt_new = ecef_to_llh(x, y, z)

    assert np.isclose(lat_orig, lat_new, atol=1e-6), "Lat conversion failed"
    assert np.isclose(lon_orig, lon_new, atol=1e-6), "Lon conversion failed"
    assert np.isclose(alt_orig, alt_new, atol=1e-3), "Alt conversion failed"
    print("  -> WGS84 <-> ECEF conversion passed.")

    # Test Haversine
    dist = haversine_distance(0, 0, 0, 1)  # 1 degree longitude at equator ~ 111km
    assert 110000 < dist < 112000, "Haversine distance calculation incorrect"
    print("  -> Haversine distance passed.")

    # ---------------------------------------------------------
    # 3. Data Loading Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Data Loading...")

    # Load Train Metadata
    train_meta = load_metadata("train")
    assert not train_meta.empty, "Train metadata is empty"
    print(f"  -> Loaded {len(train_meta)} rows from train metadata.")

    # Pick a sample trip to load raw data
    sample_row = train_meta.iloc[0]
    drive_id = sample_row["drive_id"]
    phone_name = sample_row["phone_name"]
    gnss_path = sample_row["gnss_path"]
    imu_path = sample_row["imu_path"]

    print(f"  -> Loading drive data for: {drive_id} ({phone_name})")
    # Force load from source (load_cached_data=False) to verify raw reading
    data = load_drive_data(
        drive_id, phone_name, gnss_path, imu_path, load_cached_data=False
    )

    assert "gnss" in data and not data["gnss"].empty, "GNSS data load failed"
    assert (
        "imu" in data
    ), "IMU data load failed"  # IMU might be empty depending on device, but key should exist
    print("  -> Raw data loading passed.")

    # ---------------------------------------------------------
    # 4. Feature Engineering Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying Feature Engineering...")

    # Compute features for the sample drive
    features = compute_geometric_features(
        drive_id, phone_name, gnss_path, imu_path, load_cached_data=False
    )

    # Features might be empty if no valid signals, but usually valid for train data
    if not features.empty:
        required_cols = ["UnixTimeMillis", "NetForce_E", "NetForce_N"]
        for col in required_cols:
            assert col in features.columns, f"Missing feature column: {col}"
        print(f"  -> Computed features shape: {features.shape}")
    else:
        print(
            "  -> Warning: No features computed for sample drive (could be filtering)."
        )

    # ---------------------------------------------------------
    # 5. Kinematics Verification
    # ---------------------------------------------------------
    print("\n[5] Verifying Kinematics (Odometry)...")

    odom = compute_tdcp_odometry(
        drive_id, phone_name, gnss_path, imu_path, load_cached_data=False
    )

    if not odom.empty:
        required_cols = ["UnixTimeMillis", "dE", "dN", "dU", "weight_odom"]
        for col in required_cols:
            assert col in odom.columns, f"Missing odometry column: {col}"
        print(f"  -> Computed odometry shape: {odom.shape}")
    else:
        print("  -> Warning: No odometry computed for sample drive.")

    # ---------------------------------------------------------
    # 6. Model Training Demonstration
    # ---------------------------------------------------------
    print("\n[6] Demonstrating Model Training...")

    # This function handles dataset preparation (using Config.DEBUG limits) and training
    # We use load_cached_data=False to ensure the pipeline runs from scratch
    try:
        train_models(load_cached_data=False)

        # Verify models were saved
        model_dir = os.path.join(Config.WORKING_DIR, "models")
        model_files = [f for f in os.listdir(model_dir) if f.endswith(".txt")]
        assert len(model_files) > 0, "No model files found after training"
        print(f"  -> Training successful. Models created: {len(model_files)}")
    except Exception as e:
        print(f"  -> Model training failed: {e}")
        raise e

    # ---------------------------------------------------------
    # 7. Inference and Optimization Demonstration
    # ---------------------------------------------------------
    print("\n[7] Demonstrating Inference and Global Optimization...")

    # Step 1: Generate initial predictions (ML Anchors)
    generate_submission(load_cached_data=False)

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file not created"

    sub_df = pd.read_csv(submission_path)
    print(f"  -> Initial submission generated with {len(sub_df)} rows.")

    # Step 2: Run Global Graph Optimization
    # This refines the submission using kinematics
    final_df = run_global_optimization(load_cached_data=False)

    assert not final_df.empty, "Final optimization returned empty dataframe"
    assert "LatitudeDegrees" in final_df.columns
    assert "LongitudeDegrees" in final_df.columns

    print(f"  -> Optimization complete. Final shape: {final_df.shape}")
    print("\n### Demonstration Completed Successfully ###")


if __name__ == "__main__":
    run_demonstration()
