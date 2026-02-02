import os
import shutil
import numpy as np
import pandas as pd
import warnings
import xgboost as xgb

# Import library modules
from library.config import Config
from library.utils import haversine_array, rotate_coordinates, encode_geohash
from library.pipeline import run_training_pipeline, run_inference_pipeline, set_seed
from library.model import TaxiFareXGBoost

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Creates a temporary environment with a subset of data to ensure the demo
    runs quickly (under 27 mins) and doesn't overwrite existing work.
    """
    print("Setting up demo environment...")

    # Define paths
    base_dir = os.getcwd()
    demo_metadata_dir = os.path.join(base_dir, "demo_metadata")
    demo_working_dir = os.path.join(base_dir, "demo_working")
    demo_submission_dir = os.path.join(base_dir, "demo_submission")

    # Clean up previous runs if they exist
    for d in [demo_metadata_dir, demo_working_dir, demo_submission_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # --- Create Data Subsets ---
    # Reading full parquet files is fast, but processing 44M rows in feature engineering is slow.
    # We will sample the original metadata files to create a "mini" dataset.

    print("Creating data subsets for rapid demonstration...")

    # 1. Train Set (Sample 50,000 rows)
    train_path = os.path.join(Config.METADATA_DIR, "train.parquet")
    if os.path.exists(train_path):
        # Read a small chunk
        df_train = pd.read_parquet(train_path).iloc[:50000]
        df_train.to_parquet(
            os.path.join(demo_metadata_dir, "train.parquet"), index=False
        )
    else:
        raise FileNotFoundError(f"Original train data not found at {train_path}")

    # 2. Validation Set (Sample 10,000 rows)
    val_path = os.path.join(Config.METADATA_DIR, "val.parquet")
    if os.path.exists(val_path):
        df_val = pd.read_parquet(val_path).iloc[:10000]
        df_val.to_parquet(os.path.join(demo_metadata_dir, "val.parquet"), index=False)
    else:
        raise FileNotFoundError(f"Original val data not found at {val_path}")

    # 3. Test Set (Copy all, it's small)
    test_path = os.path.join(Config.METADATA_DIR, "test.parquet")
    if os.path.exists(test_path):
        df_test = pd.read_parquet(test_path)
        df_test.to_parquet(os.path.join(demo_metadata_dir, "test.parquet"), index=False)
    else:
        raise FileNotFoundError(f"Original test data not found at {test_path}")

    # --- Override Config ---
    print("Overriding configuration for demo...")
    Config.METADATA_DIR = demo_metadata_dir
    Config.WORKING_DIR = demo_working_dir
    Config.SUBMISSION_DIR = demo_submission_dir
    Config.SUBMISSION_PATH = os.path.join(demo_submission_dir, "submission.csv")

    # Reduce computational load
    Config.NUM_FOLDS = 2  # Reduce folds for feature engineering subtraction

    return demo_submission_dir


def verify_utils():
    """
    Verifies the correctness of utility functions (Distance, Rotation, Geohashing).
    """
    print("\n--- Verifying Utility Functions ---")

    # 1. Test Haversine Distance
    # Distance between (0, 0) and (1, 0) degrees latitude is approx 111.195 km
    lat1, lon1 = np.array([0.0]), np.array([0.0])
    lat2, lon2 = np.array([1.0]), np.array([0.0])

    dist = haversine_array(lat1, lon1, lat2, lon2)[0]
    expected_dist = 111.19
    assert np.isclose(
        dist, expected_dist, atol=0.5
    ), f"Haversine calculation failed. Got {dist}, expected ~{expected_dist}"
    print("Haversine Distance: OK")

    # 2. Test Coordinate Rotation
    # Rotate (lat=0, lon=1) by 90 degrees.
    # x (lon) = 1, y (lat) = 0.
    # x' = x cos(90) - y sin(90) = 1*0 - 0*1 = 0
    # y' = x sin(90) + y cos(90) = 1*1 + 0*0 = 1
    # Result should be lat=1, lon=0
    lat, lon = 0.0, 1.0
    lat_rot, lon_rot = rotate_coordinates(lat, lon, 90)
    assert np.isclose(lat_rot, 1.0) and np.isclose(
        lon_rot, 0.0
    ), f"Rotation failed. Got lat={lat_rot}, lon={lon_rot}"
    print("Coordinate Rotation: OK")

    # 3. Test Geohashing
    # NYC Coordinates: 40.7128° N, 74.0060° W
    # Note: Library expects negative longitude for West.
    lat_nyc = 40.7128
    lon_nyc = -74.0060
    # Precision 5 geohash for NYC
    gh = encode_geohash([lat_nyc], [lon_nyc], precision=5)[0]
    # 'dr5re' is a known geohash prefix for NYC area
    assert gh.startswith(
        "dr5"
    ), f"Geohashing failed. Got {gh}, expected start with 'dr5'"
    print("Geohashing: OK")


def demonstrate_pipeline():
    """
    Runs the training and inference pipeline using the library functions.
    """
    print("\n--- Starting Pipeline Demonstration ---")

    # Ensure reproducibility
    set_seed(42)

    # 1. Run Training Pipeline
    # We override hyperparameters to make training instant:
    # - learner_sample_size: Use all available in the small subset (up to 50k)
    # - n_estimators: 10 trees (enough to verify flow)
    # - early_stopping_rounds: 5
    # - load_cached_data: False (Force feature engineering execution to test logic)
    print("Executing Training Pipeline...")
    model_wrapper, test_feat_df = run_training_pipeline(
        load_cached_data=False,
        learner_sample_size=10000,
        n_estimators=10,
        early_stopping_rounds=5,
    )

    # Verification of Training Output
    assert isinstance(
        model_wrapper, TaxiFareXGBoost
    ), "Pipeline did not return a TaxiFareXGBoost instance."
    assert hasattr(
        model_wrapper.model, "predict"
    ), "Inner model is not a valid estimator."

    # Verification of Feature Engineering Output
    print("Verifying Feature Engineering...")
    expected_cols_subset = [
        "pickup_latitude",
        "pickup_longitude",
        "geo_L5_mean",
        "geo_L6_mean",
        "geo_L7_mean",  # From GeohashTargetEncoder
        "dist_haversine",
        "dist_manhattan",  # From add_geometric_features
        "pickup_lat_rot",  # From rotation
    ]

    for col in expected_cols_subset:
        assert col in test_feat_df.columns, f"Missing feature column: {col}"

    assert (
        "fare_amount" not in test_feat_df.columns
    ), "Leakage: Target variable found in test features."
    print("Feature Engineering: OK")

    # 2. Run Inference Pipeline
    print("Executing Inference Pipeline...")
    run_inference_pipeline(model_wrapper, test_feat_df)

    # Verification of Submission
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    submission_df = pd.read_csv(submission_path)

    # Check format
    assert list(submission_df.columns) == [
        "key",
        "fare_amount",
    ], "Submission columns incorrect."
    assert len(submission_df) == len(test_feat_df), "Submission row count mismatch."
    assert (
        submission_df["fare_amount"].min() >= 2.50
    ), "Post-processing min fare floor failed."

    print(f"Submission generated successfully with {len(submission_df)} rows.")
    print(f"Sample predictions:\n{submission_df.head(3)}")


if __name__ == "__main__":
    try:
        # 1. Setup
        setup_demo_environment()

        # 2. Verify Utils
        verify_utils()

        # 3. Run Pipeline
        demonstrate_pipeline()

        print("\nSUCCESS: All demonstrations and verifications passed.")

    except AssertionError as e:
        print(f"\nFAILURE: Assertion failed - {e}")
        exit(1)
    except Exception as e:
        print(f"\nFAILURE: An unexpected error occurred - {e}")
        # Print traceback for debugging if needed, but error message is usually enough
        import traceback

        traceback.print_exc()
        exit(1)
