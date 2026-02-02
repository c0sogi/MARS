import os
import pandas as pd
import numpy as np
import shutil
import warnings

# Import library modules
import library.utils as utils
import library.data_loader as data_loader
import library.training as training
import library.inference as inference
import library.model as model_lib

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
ORIGINAL_METADATA_DIR = "./metadata"
DEBUG_METADATA_DIR = "./working/metadata_debug"
CACHE_DIR = "./working/idea_12"
MODELS_DIR = os.path.join(CACHE_DIR, "models")
SUBMISSION_PATH = "./submission/submission.csv"
SEED = 42


def setup_debug_environment():
    """Creates a subset of metadata for quick demonstration."""
    print("Setting up debug environment...")
    os.makedirs(DEBUG_METADATA_DIR, exist_ok=True)

    # 1. Create subset of Train Metadata
    train_meta_path = os.path.join(ORIGINAL_METADATA_DIR, "train_metadata.csv")
    if os.path.exists(train_meta_path):
        df_train = pd.read_csv(train_meta_path)
        # Sample 2 unique drives
        unique_drives = df_train["drive_id"].unique()
        if len(unique_drives) > 2:
            sampled_drives = unique_drives[:2]
            df_train_sub = df_train[df_train["drive_id"].isin(sampled_drives)]
        else:
            df_train_sub = df_train

        # Save to debug dir
        df_train_sub.to_csv(
            os.path.join(DEBUG_METADATA_DIR, "train_metadata.csv"), index=False
        )
        print(
            f"Created debug train metadata with {len(df_train_sub)} rows (Drives: {df_train_sub['drive_id'].unique()})"
        )
    else:
        raise FileNotFoundError(
            f"Original train metadata not found at {train_meta_path}"
        )

    # 2. Create subset of Test Metadata
    test_meta_path = os.path.join(ORIGINAL_METADATA_DIR, "test_metadata.csv")
    if os.path.exists(test_meta_path):
        df_test = pd.read_csv(test_meta_path)
        # Sample 2 unique trips
        unique_trips = df_test["tripId"].unique()
        if len(unique_trips) > 2:
            sampled_trips = unique_trips[:2]
            df_test_sub = df_test[df_test["tripId"].isin(sampled_trips)]
        else:
            df_test_sub = df_test

        # Save to debug dir
        df_test_sub.to_csv(
            os.path.join(DEBUG_METADATA_DIR, "test_metadata.csv"), index=False
        )
        print(
            f"Created debug test metadata with {len(df_test_sub)} rows (Trips: {df_test_sub['tripId'].unique()})"
        )
    else:
        raise FileNotFoundError(f"Original test metadata not found at {test_meta_path}")

    # 3. Patch library constants to use debug metadata
    # This redirects data loading to our small subset
    data_loader.METADATA_DIR = DEBUG_METADATA_DIR
    training.METADATA_DIR = DEBUG_METADATA_DIR

    # Clear cache to ensure we don't load old full datasets
    if os.path.exists(CACHE_DIR):
        print(f"Clearing cache directory: {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)


def verify_utils():
    """Demonstrates and verifies utility functions."""
    print("\n--- Verifying Utils ---")

    # Test 1: Coordinate Conversion Round-trip
    lat, lon, alt = 37.42, -122.08, 30.0
    x, y, z = utils.wgs84_to_ecef(lat, lon, alt)
    lat_out, lon_out, alt_out = utils.ecef_to_wgs84(x, y, z)

    print(f"Original LLA: {lat}, {lon}, {alt}")
    print(f"Converted ECEF: {x:.2f}, {y:.2f}, {z:.2f}")
    print(f"Restored LLA: {lat_out:.6f}, {lon_out:.6f}, {alt_out:.6f}")

    np.testing.assert_almost_equal(
        lat, lat_out, decimal=5, err_msg="Latitude conversion failed"
    )
    np.testing.assert_almost_equal(
        lon, lon_out, decimal=5, err_msg="Longitude conversion failed"
    )
    np.testing.assert_almost_equal(
        alt, alt_out, decimal=3, err_msg="Altitude conversion failed"
    )
    print("Coordinate conversion round-trip passed.")

    # Test 2: Haversine Distance
    # Distance between (0,0) and (0, 1) degree at equator should be approx 111km
    d = utils.haversine_distance(0, 0, 0, 1)
    print(f"Haversine distance (0,0) to (0,1): {d:.2f} meters")
    assert 111000 < d < 112000, "Haversine distance calculation seems off"
    print("Haversine distance check passed.")


def run_training_demo():
    """Runs the training pipeline on the debug dataset."""
    print("\n--- Running Training Demo ---")

    # Run training with 2 folds for speed
    # load_cached_data=False ensures we process our new debug metadata
    model = training.run_group_kfold(
        n_folds=2, load_cached_data=False, debug=False, seed=SEED
    )

    # Verify model object
    assert isinstance(
        model, model_lib.ResidualRegressor
    ), "Training did not return a ResidualRegressor object"
    assert len(model.models_e) == 2, "Model should have 2 East boosters"
    assert len(model.models_n) == 2, "Model should have 2 North boosters"

    # Verify artifacts
    assert os.path.exists(MODELS_DIR), "Models directory not created"
    assert os.path.exists(
        os.path.join(MODELS_DIR, "lgbm_east_fold_0.txt")
    ), "East model fold 0 not saved"

    print("Training demo completed successfully.")


def run_inference_demo():
    """Runs the inference pipeline on the debug dataset."""
    print("\n--- Running Inference Demo ---")

    # Generate submission
    # load_cached_data=False ensures we process the debug test metadata
    inference.generate_submission(load_cached_data=False)

    # Verify submission file
    assert os.path.exists(SUBMISSION_PATH), "Submission file not created"

    df_sub = pd.read_csv(SUBMISSION_PATH)
    print(f"Submission generated with shape: {df_sub.shape}")

    # Verify columns
    expected_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    assert (
        list(df_sub.columns) == expected_cols
    ), "Submission columns do not match requirements"

    # Verify no NaNs (inference logic handles fallback)
    assert not df_sub.isnull().any().any(), "Submission contains NaNs"

    print("Inference demo completed successfully.")


if __name__ == "__main__":
    # 1. Setup Environment (Subset Data & Patching)
    setup_debug_environment()

    # 2. Verify Utility Functions
    verify_utils()

    # 3. Run Training Pipeline
    run_training_demo()

    # 4. Run Inference Pipeline
    run_inference_demo()

    print("\nAll demonstrations passed!")
