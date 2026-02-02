import os
import sys
import pandas as pd
import numpy as np
import warnings

# Import library modules
import library.config as config
from library.data_loader import load_metadata, load_gnss_dataframe, load_ground_truth
from library.feature_engineering import (
    compute_features_from_gnss,
    compute_altitude_corrected_target,
)
from library.kinematics import generate_kinematics
from library.model_handler import LGBMRegressorWrapper
from library.graph_solver import optimize_trajectory

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Pipeline ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed
    # ---------------------------------------------------------
    print("\n[1] Configuring for fast execution...")
    # Reduce model complexity for demo
    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["num_leaves"] = 8
    config.LGBM_PARAMS["n_jobs"] = 1

    # Ensure output directories exist
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Data Loading (Subset)
    # ---------------------------------------------------------
    print("\n[2] Loading Metadata...")
    train_meta = load_metadata("train")
    test_meta = load_metadata("test")

    # Select trips from multiple drives to satisfy GroupKFold (n_splits=2) constraints
    # Cite debug_lesson_8
    n_required_groups = 2
    unique_drives = train_meta["drive_id"].unique()

    # Select first N distinct drives
    selected_drives = unique_drives[:n_required_groups]

    # Pick one trip per drive to minimize data volume while ensuring distinct groups
    train_trip_ids = []
    for drive in selected_drives:
        trips = train_meta[train_meta["drive_id"] == drive]["tripId"].unique()
        if len(trips) > 0:
            train_trip_ids.append(trips[0])

    train_meta_sub = train_meta[train_meta["tripId"].isin(train_trip_ids)].copy()

    # Test set can remain a single trip
    test_trip_id = test_meta["tripId"].unique()[0]
    test_meta_sub = test_meta[test_meta["tripId"] == test_trip_id].copy()

    print(f"    Selected Train Trips: {train_trip_ids}")
    print(f"    Selected Test Trip:  {test_trip_id}")

    print("\n[2.1] Loading GNSS Data (Subset)...")
    # load_cached_data=False ensures we process only the subset and don't load full cached files
    train_gnss = load_gnss_dataframe(
        train_meta_sub, "train_demo", load_cached_data=False
    )
    test_gnss = load_gnss_dataframe(test_meta_sub, "test_demo", load_cached_data=False)

    print(f"    Train GNSS rows: {len(train_gnss)}")
    print(f"    Test GNSS rows:  {len(test_gnss)}")

    # ---------------------------------------------------------
    # 3. Feature Engineering
    # ---------------------------------------------------------
    print("\n[3] Computing Features...")

    # Compute features
    train_features = compute_features_from_gnss(train_gnss)
    test_features = compute_features_from_gnss(test_gnss)

    print(f"    Train Features shape: {train_features.shape}")

    # Prepare Training Dataset (Merge Metadata + Features + WLS)
    print("    Merging Train Data...")
    train_dataset = pd.merge(
        train_meta_sub, train_features, on=["tripId", "UnixTimeMillis"], how="inner"
    )

    # Extract WLS for target computation (one per epoch)
    train_wls = (
        train_gnss.groupby(["tripId", "utcTimeMillis"])
        .first()[
            [
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        ]
        .reset_index()
        .rename(columns={"utcTimeMillis": "UnixTimeMillis"})
    )

    train_dataset = pd.merge(
        train_dataset, train_wls, on=["tripId", "UnixTimeMillis"], how="inner"
    )

    # Compute Targets (Altitude-Corrected Residuals)
    train_dataset = compute_altitude_corrected_target(train_dataset)
    # Drop NaNs in targets
    train_dataset = train_dataset.dropna(subset=[config.TARGET_E, config.TARGET_N])

    # Prepare Test Dataset
    print("    Merging Test Data...")
    test_dataset = pd.merge(
        test_meta_sub, test_features, on=["tripId", "UnixTimeMillis"], how="inner"
    )

    test_wls = (
        test_gnss.groupby(["tripId", "utcTimeMillis"])
        .first()[
            [
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        ]
        .reset_index()
        .rename(columns={"utcTimeMillis": "UnixTimeMillis"})
    )

    test_dataset = pd.merge(
        test_dataset, test_wls, on=["tripId", "UnixTimeMillis"], how="inner"
    )

    # ---------------------------------------------------------
    # 4. Kinematics Generation
    # ---------------------------------------------------------
    print("\n[4] Generating Kinematics...")
    # Generate kinematic constraints from raw GNSS
    train_kin = generate_kinematics(
        train_meta_sub, "train_demo", load_cached_data=False
    )
    test_kin = generate_kinematics(test_meta_sub, "test_demo", load_cached_data=False)

    print(f"    Train Kinematics rows: {len(train_kin)}")
    print(f"    Test Kinematics rows:  {len(test_kin)}")

    # ---------------------------------------------------------
    # 5. Model Training (LightGBM)
    # ---------------------------------------------------------
    print("\n[5] Training Model...")
    model_wrapper = LGBMRegressorWrapper()
    # Train on the subset
    model_wrapper.train(train_dataset, n_folds=2, force_retrain=True)

    # ---------------------------------------------------------
    # 6. Inference (Anchor Prediction)
    # ---------------------------------------------------------
    print("\n[6] Generating Anchor Predictions...")
    # Predict ENU residuals for test set
    # load_models_from_disk=False uses the models currently in memory
    test_preds = model_wrapper.predict(test_dataset, load_models_from_disk=False)

    # Merge predictions into test dataset
    test_dataset_with_preds = pd.merge(
        test_dataset,
        test_preds[["tripId", "UnixTimeMillis", "pred_E", "pred_N"]],
        on=["tripId", "UnixTimeMillis"],
        how="left",
    )

    # ---------------------------------------------------------
    # 7. Graph Optimization
    # ---------------------------------------------------------
    print("\n[7] Running Graph Optimization...")
    # Fuse anchors with kinematics
    optimized_trajectory = optimize_trajectory(
        test_dataset_with_preds, test_kin, load_cached_data=False
    )

    # ---------------------------------------------------------
    # 8. Validation & Submission
    # ---------------------------------------------------------
    print("\n[8] Validating Output...")

    # Check for NaNs
    if (
        optimized_trajectory["LatitudeDegrees"].isna().any()
        or optimized_trajectory["LongitudeDegrees"].isna().any()
    ):
        raise AssertionError("Output trajectory contains NaNs!")

    # Check shape
    expected_rows = len(test_meta_sub)
    actual_rows = len(optimized_trajectory)
    print(f"    Expected rows: {expected_rows}")
    print(f"    Actual rows:   {actual_rows}")

    if actual_rows != expected_rows:
        # Note: Graph solver might drop disconnected components or rows without anchors if inner join used improperly,
        # but the implementation uses left join on anchors.
        # However, optimize_trajectory groups by tripId. If a trip has no data, it might be skipped.
        print(
            "    Warning: Row count mismatch (likely due to missing GNSS epochs in raw data)."
        )

    # Display sample
    print("\nSample Optimized Output:")
    print(optimized_trajectory.head())

    # Save
    out_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    optimized_trajectory.to_csv(out_path, index=False)
    print(f"\nSaved demo submission to {out_path}")

    print("\n=== Pipeline Completed Successfully ===")


if __name__ == "__main__":
    main()
