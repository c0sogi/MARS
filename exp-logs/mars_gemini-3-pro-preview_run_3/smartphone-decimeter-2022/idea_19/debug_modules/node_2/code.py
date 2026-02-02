import sys
import os
import pandas as pd
import numpy as np
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

# Import library modules
import library.config as config
from library.data_manager import DataManager
from library.model import SplitBandLGBM
from library.utils import ecef_to_wgs84

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def validate_feature_generation(df):
    """
    Validates that the feature engineering pipeline produced expected columns.
    """
    print("\n[Validation] Checking feature columns...")

    # Check for split-band projection features
    l1_force_cols = [c for c in df.columns if "L1_PrForce" in c]
    print(f"  Found {len(l1_force_cols)} L1 Projection Force columns.")
    if len(l1_force_cols) == 0:
        raise AssertionError("Failed to generate L1 Projection features.")

    # Check for kinematic features
    if "Doppler_Vel_X" in df.columns:
        print("  Doppler Velocity features present.")
    else:
        print(
            "  Warning: Doppler Velocity features missing (possibly due to data quality in subset)."
        )

    # Check for WLS baseline
    if "WlsPositionXEcefMeters" not in df.columns:
        raise AssertionError("WLS Baseline columns missing from features.")

    print("  Feature validation passed.")


def main():
    print("=== Starting Solution Demonstration ===\n")

    # 1. Configure for Speed/Demo
    # We override configuration parameters to ensure the script runs quickly
    print("[Config] Overriding configuration for demonstration...")
    # Disable random sampling in DataManager to allow manual contiguous sampling
    config.DEBUG = False
    # config.DEBUG_SAMPLE_SIZE = 100  # Not used when DEBUG is False

    # Reduce model complexity for fast training
    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["min_child_samples"] = 5
    config.LGBM_PARAMS["num_leaves"] = 8

    # 2. Initialize Data Manager
    print("\n[Data] Initializing DataManager...")
    dm = DataManager()

    # 3. Load Metadata
    print("[Data] Loading Train/Val Metadata...")
    train_meta, val_meta = dm.load_train_val_metadata()

    # Filter to multiple drives to satisfy GroupKFold(n_splits=5)
    # We need at least 5 groups. We'll take small contiguous chunks from each.
    if not train_meta.empty:
        unique_drives = train_meta["drive_id"].unique()
        # Ensure we have at least 5 drives
        n_drives = 5
        if len(unique_drives) < n_drives:
            print(f"Warning: Only {len(unique_drives)} drives available. Using all.")
            selected_drives = unique_drives
        else:
            selected_drives = unique_drives[:n_drives]

        print(f"  Selecting subsets from Drives: {selected_drives}")

        subsets = []
        rows_per_drive = 20  # 5 drives * 20 rows = 100 rows total

        for drive in selected_drives:
            # Get data for this drive
            drive_data = train_meta[train_meta["drive_id"] == drive]
            if drive_data.empty:
                continue

            # Pick first phone
            phone = drive_data["phone_name"].iloc[0]

            # Select first N contiguous epochs
            subset = (
                drive_data[drive_data["phone_name"] == phone]
                .sort_values("UnixTimeMillis")
                .head(rows_per_drive)
                .copy()
            )
            subsets.append(subset)

        if not subsets:
            raise ValueError("Failed to create training subsets.")

        train_meta_subset = pd.concat(subsets)
    else:
        raise ValueError("Train metadata is empty.")

    # 4. Feature Engineering (Train)
    print(
        f"\n[Features] Processing Training Dataset ({len(train_meta_subset)} rows)..."
    )
    # load_cached_data=False forces re-computation to demonstrate the logic
    train_df = dm.prepare_dataset(train_meta_subset, "train", load_cached_data=False)

    validate_feature_generation(train_df)

    # 5. Model Training
    print("\n[Model] Training Split-Band LightGBM...")
    model = SplitBandLGBM()
    model.train(train_df)

    # 6. Test / Inference Preparation
    print("\n[Inference] Loading Test Metadata...")
    test_meta = dm.load_test_metadata()

    # Select a subset for test inference (contiguous trip)
    if not test_meta.empty:
        drive_id_test = test_meta["drive_id"].iloc[0]
        phone_name_test = test_meta["phone_name"].iloc[0]
        print(
            f"  Selecting subset from Test Drive: {drive_id_test}, Phone: {phone_name_test}"
        )

        test_meta_subset = (
            test_meta[
                (test_meta["drive_id"] == drive_id_test)
                & (test_meta["phone_name"] == phone_name_test)
            ]
            .sort_values("UnixTimeMillis")
            .head(100)
            .copy()
        )
    else:
        raise ValueError("Test metadata is empty.")

    print(f"[Features] Processing Test Dataset ({len(test_meta_subset)} rows)...")
    test_df = dm.prepare_dataset(test_meta_subset, "test", load_cached_data=False)

    # 7. Prediction & Optimization
    print("\n[Inference] Running Prediction and Graph Optimization...")
    # This step predicts residuals using LGBM and then refines trajectory using the Graph Optimizer
    submission = model.predict(test_df)

    # 8. Final Validation
    print("\n[Validation] Verifying Submission...")

    # Check output file
    if not os.path.exists(config.SUBMISSION_FILE_PATH):
        raise AssertionError("Submission file was not created.")

    # Check columns
    required_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    if not all(col in submission.columns for col in required_cols):
        raise AssertionError(f"Submission missing required columns: {required_cols}")

    # Check values
    lat_min, lat_max = (
        submission["LatitudeDegrees"].min(),
        submission["LatitudeDegrees"].max(),
    )
    lon_min, lon_max = (
        submission["LongitudeDegrees"].min(),
        submission["LongitudeDegrees"].max(),
    )

    print(f"  Latitude Range: [{lat_min:.4f}, {lat_max:.4f}]")
    print(f"  Longitude Range: [{lon_min:.4f}, {lon_max:.4f}]")

    if not (-90 <= lat_min and lat_max <= 90):
        raise AssertionError("Latitude values out of bounds.")
    if not (-180 <= lon_min and lon_max <= 180):
        raise AssertionError("Longitude values out of bounds.")

    # Check that we actually updated values compared to WLS (Baseline)
    # We extract WLS from test_df and convert to Lat/Lon to compare with optimized result
    print("\n[Validation] Comparing Optimized Result vs WLS Baseline...")

    # Join WLS from test_df to submission subset
    # Note: submission contains all rows from sample_submission, test_df is our subset
    # We filter submission to our subset for comparison
    sub_subset = submission.merge(
        test_df[
            [
                "tripId",
                "UnixTimeMillis",
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        ],
        on=["tripId", "UnixTimeMillis"],
        how="inner",
    )

    if not sub_subset.empty:
        # Convert WLS ECEF to Lat/Lon
        wls_lat, wls_lon, _ = ecef_to_wgs84(
            sub_subset["WlsPositionXEcefMeters"].values,
            sub_subset["WlsPositionYEcefMeters"].values,
            sub_subset["WlsPositionZEcefMeters"].values,
        )

        diff_lat = np.abs(sub_subset["LatitudeDegrees"] - wls_lat)
        diff_lon = np.abs(sub_subset["LongitudeDegrees"] - wls_lon)

        mean_diff = np.mean(diff_lat + diff_lon)
        print(f"  Mean deviation from WLS Baseline: {mean_diff:.8f} degrees")

        if mean_diff < 1e-9:
            print(
                "  Warning: Optimization result is identical to WLS. Model might be predicting zero residuals."
            )
        else:
            print("  Success: Optimization modified the trajectory.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
