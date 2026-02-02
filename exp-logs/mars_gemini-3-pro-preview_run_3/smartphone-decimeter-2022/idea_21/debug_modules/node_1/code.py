import os
import sys
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure the current directory is in the python path for library imports
sys.path.append(os.getcwd())

# Import from the provided library
from library.config import INPUT_DIR, METADATA_DIR, WORKING_DIR
from library.data_loader import load_drive_data
from library.feature_eng import (
    compute_satellite_residuals,
    project_forces,
    generate_features,
    create_dataset,
)
from library.kinematics import compute_trajectory_deltas
from library.model import LGBMResidualPredictor
from library.optimizer import TrajectoryOptimizer


def test_data_loader(drive_id, phone_name, metadata_df):
    print("\n[1] Testing Data Loader...")

    # Load data for specific drive
    # We set load_cached_data=False to verify the loading logic from raw files
    aligned_gnss, raw_imu = load_drive_data(
        drive_id, phone_name, metadata_df, load_cached_data=False
    )

    print(f"    Aligned GNSS Shape: {aligned_gnss.shape}")
    print(f"    Raw IMU Shape: {raw_imu.shape}")

    # Assertions
    assert not aligned_gnss.empty, "Aligned GNSS dataframe is empty"
    assert (
        "UnixTimeMillis" in aligned_gnss.columns
    ), "Missing UnixTimeMillis in aligned GNSS"
    assert (
        "WlsPositionXEcefMeters" in aligned_gnss.columns
    ), "Missing WLS Position in aligned GNSS"

    # Check alignment
    # Ground truth usually has 1Hz data.
    # Check if timestamps are roughly 1000ms apart (allowing for gaps)
    timestamps = aligned_gnss["UnixTimeMillis"].unique()
    if len(timestamps) > 1:
        diffs = np.diff(np.sort(timestamps))
        # Most diffs should be multiples of 1000
        print(f"    Mean Time Diff: {np.mean(diffs):.2f} ms")

    return aligned_gnss


def test_feature_engineering(aligned_gnss, drive_id, phone_name, metadata_df):
    print("\n[2] Testing Feature Engineering...")

    # Step 1: Residuals
    gnss_res = compute_satellite_residuals(aligned_gnss)
    print(f"    GNSS with Residuals Shape: {gnss_res.shape}")
    assert "pr_residual" in gnss_res.columns, "Pseudorange residual missing"

    # Step 2: Project Forces
    gnss_proj = project_forces(gnss_res)
    print(f"    Projected Forces (Aggregated) Shape: {gnss_proj.shape}")
    assert "F_L1_E" in gnss_proj.columns, "Projected force F_L1_E missing"

    # Step 3: Full Feature Generation Wrapper
    # Use a subset of metadata to simulate dataset creation
    subset_meta = metadata_df[
        (metadata_df["drive_id"] == drive_id)
        & (metadata_df["phone_name"] == phone_name)
    ]

    features_df = create_dataset(subset_meta, load_cached_data=False)
    print(f"    Generated Features DF Shape: {features_df.shape}")

    # Verify Targets exist (since we are using training data)
    assert "Target_E" in features_df.columns, "Target_E missing from features"
    assert "Target_N" in features_df.columns, "Target_N missing from features"

    return features_df


def test_kinematics(drive_id, phone_name, aligned_gnss):
    print("\n[3] Testing Kinematics...")

    # Compute deltas
    deltas_df = compute_trajectory_deltas(
        drive_id, phone_name, aligned_gnss, load_cached_data=False
    )

    print(f"    Kinematics Deltas Shape: {deltas_df.shape}")
    print(f"    Columns: {list(deltas_df.columns)}")

    assert not deltas_df.empty, "Kinematics dataframe is empty"
    assert "dx" in deltas_df.columns, "dx missing"
    assert "weight" in deltas_df.columns, "weight missing"

    return deltas_df


def test_modeling(features_df):
    print("\n[4] Testing Model Training & Prediction...")

    # Initialize Model
    predictor = LGBMResidualPredictor()

    # REDUCE COMPLEXITY FOR DEMO SPEED
    predictor.params["n_estimators"] = 10
    predictor.params["num_leaves"] = 8

    # Split into train/val (using the same drive for demo purposes, usually disjoint)
    # In practice, we need different drives. Here we just split the dataframe.
    split_idx = int(len(features_df) * 0.8)
    train_df = features_df.iloc[:split_idx].copy()
    val_df = features_df.iloc[split_idx:].copy()

    # Train
    print("    Training model on subset...")
    predictor.train(train_df, val_df)

    # Predict
    print("    Predicting on validation subset...")
    preds = predictor.predict(val_df)

    print(f"    Predictions Shape: {preds.shape}")
    print(f"    Prediction Columns: {list(preds.columns)}")

    assert "pred_E" in preds.columns
    assert "pred_N" in preds.columns
    assert len(preds) == len(val_df)

    return predictor, preds


def test_optimization(drive_id, phone_name, metadata_df, features_df, predictor):
    print("\n[5] Testing Trajectory Optimization...")

    # We need predictions for the full drive to optimize it
    full_preds = predictor.predict(features_df)

    optimizer = TrajectoryOptimizer()

    # Run optimization
    # Note: load_cached_data=False to force execution
    optimized_df = optimizer.optimize_drive(
        drive_id, phone_name, metadata_df, full_preds, load_cached_data=False
    )

    print(f"    Optimized Trajectory Shape: {optimized_df.shape}")

    if not optimized_df.empty:
        print(f"    Sample Result:\n{optimized_df.head(2)}")

        # Validation
        assert "LatitudeDegrees" in optimized_df.columns
        assert "LongitudeDegrees" in optimized_df.columns

        # Check if coordinates are valid (roughly within US range for this dataset)
        lat = optimized_df["LatitudeDegrees"].mean()
        lon = optimized_df["LongitudeDegrees"].mean()
        print(f"    Mean Lat/Lon: {lat:.4f}, {lon:.4f}")

        assert 30 < lat < 45, f"Latitude {lat} seems out of range for US dataset"
        assert -130 < lon < -70, f"Longitude {lon} seems out of range for US dataset"
    else:
        print(
            "    Warning: Optimization returned empty dataframe (possibly no overlapping timestamps)."
        )


if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)

    # 1. Setup Metadata
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    if not os.path.exists(train_meta_path):
        print("Metadata not found. Please ensure metadata generation script has run.")
        sys.exit(1)

    df_meta = pd.read_csv(train_meta_path)

    # Pick a specific drive and phone for the demo
    # We look for a drive that definitely exists in the provided file list
    target_drive = "2020-05-15-US-MTV-1"
    target_phone = "GooglePixel4XL"

    print(f"Selected Drive: {target_drive}")
    print(f"Selected Phone: {target_phone}")

    # Filter metadata for this drive
    drive_meta = df_meta[
        (df_meta["drive_id"] == target_drive) & (df_meta["phone_name"] == target_phone)
    ]

    if drive_meta.empty:
        print(f"Drive {target_drive} not found in metadata. Picking first available.")
        target_drive = df_meta.iloc[0]["drive_id"]
        target_phone = df_meta.iloc[0]["phone_name"]
        drive_meta = df_meta[
            (df_meta["drive_id"] == target_drive)
            & (df_meta["phone_name"] == target_phone)
        ]
        print(f"Fallback Drive: {target_drive}, Phone: {target_phone}")

    # 2. Run Components
    try:
        # Data Loading
        aligned_gnss = test_data_loader(target_drive, target_phone, df_meta)

        # Feature Engineering
        features_df = test_feature_engineering(
            aligned_gnss, target_drive, target_phone, df_meta
        )

        # Kinematics
        # Note: We pass the aligned GNSS which contains raw measurements required for kinematics
        kinematics_df = test_kinematics(target_drive, target_phone, aligned_gnss)

        # Modeling
        if not features_df.empty:
            model, preds = test_modeling(features_df)

            # Optimization
            test_optimization(target_drive, target_phone, df_meta, features_df, model)
        else:
            print("Skipping modeling and optimization due to empty features.")

        print("\n=== Demo Completed Successfully ===")

    except Exception as e:
        print(f"\n!!! Demo Failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
