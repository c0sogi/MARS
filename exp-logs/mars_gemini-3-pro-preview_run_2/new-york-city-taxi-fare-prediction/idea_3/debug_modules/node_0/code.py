import os
import sys
import numpy as np
import pandas as pd
import shutil
import warnings

# Import provided library modules
from library import config
from library import feature_ops
from library.data_handler import DataHandler
from library.ensemble_learner import EnsembleLearner

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Sets up a lightweight environment for the demo by creating a subset of the data
    and overriding configuration parameters to ensure speed.
    """
    print(">>> Setting up demo environment...")

    # Define temporary directories
    demo_working_dir = "./working/demo_run"
    demo_data_dir = os.path.join(demo_working_dir, "data")

    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_data_dir, exist_ok=True)

    # 1. Create Data Subsets (Read first 10k rows to save time)
    print("    Creating data subsets (10k rows)...")

    # Paths to original metadata
    orig_train_path = "./metadata/train.parquet"
    orig_val_path = "./metadata/val.parquet"
    orig_test_path = "./metadata/test.parquet"

    # Paths for demo data
    demo_train_path = os.path.join(demo_data_dir, "train_small.parquet")
    demo_val_path = os.path.join(demo_data_dir, "val_small.parquet")
    demo_test_path = os.path.join(demo_data_dir, "test_small.parquet")

    # Read and save subsets
    # We use pandas read_parquet with a limited columns read if possible,
    # but since we need head, we just read and slice.
    # Note: read_parquet doesn't support 'nrows' directly in all engines,
    # but pyarrow backing usually handles large files well.
    # To be safe and fast, we read the file.

    pd.read_parquet(orig_train_path).head(10000).to_parquet(
        demo_train_path, index=False
    )
    pd.read_parquet(orig_val_path).head(2000).to_parquet(demo_val_path, index=False)
    pd.read_parquet(orig_test_path).head(1000).to_parquet(demo_test_path, index=False)

    # 2. Override Config
    print("    Overriding configuration for speed...")

    # Update Paths
    config.WORKING_DIR = demo_working_dir
    config.DATA_PATHS["train_parquet"] = demo_train_path
    config.DATA_PATHS["val_parquet"] = demo_val_path
    config.DATA_PATHS["test_parquet"] = demo_test_path
    config.DATA_PATHS["train_processed"] = os.path.join(
        demo_working_dir, "train_processed.parquet"
    )
    config.DATA_PATHS["val_processed"] = os.path.join(
        demo_working_dir, "val_processed.parquet"
    )
    config.DATA_PATHS["test_processed"] = os.path.join(
        demo_working_dir, "test_processed.parquet"
    )
    config.DATA_PATHS["submission"] = os.path.join(demo_working_dir, "submission.csv")

    # Update XGBoost Params for speed
    config.XGB_PARAMS["n_estimators"] = 10
    config.XGB_PARAMS["early_stopping_rounds"] = 5
    config.XGB_PARAMS["n_jobs"] = 4
    # Ensure we use a compatible device setting (CPU is safer for tiny datasets to avoid overhead)
    config.XGB_PARAMS["device"] = "cpu"
    config.XGB_PARAMS["tree_method"] = "hist"

    # Update Ensemble Config
    config.ENSEMBLE_CONFIG["n_models"] = 2

    print(">>> Environment setup complete.\n")


def test_feature_ops():
    """
    Validates the logic in library/feature_ops.py
    """
    print(">>> Testing Feature Operations...")

    # 1. Test Haversine Distance
    # Distance between (0,0) and (1,0) degrees latitude is approx 111.195 km
    lat1, lon1 = 0.0, 0.0
    lat2, lon2 = 1.0, 0.0
    dist = feature_ops.haversine_distance(lat1, lon1, lat2, lon2)

    print(f"    Haversine (0,0)->(1,0): {dist:.4f} km")
    assert np.isclose(
        dist, 111.195, atol=0.1
    ), f"Haversine calculation incorrect. Expected ~111.195, got {dist}"

    # 2. Test Manhattan Distance
    # |0-1| + |0-0| = 1.0 degree
    m_dist = feature_ops.manhattan_distance(lat1, lon1, lat2, lon2)
    print(f"    Manhattan (0,0)->(1,0): {m_dist:.4f} deg")
    assert np.isclose(
        m_dist, 1.0
    ), f"Manhattan calculation incorrect. Expected 1.0, got {m_dist}"

    # 3. Test Rotation
    # Rotate (1, 0) by 90 degrees.
    # x=0, y=1 (lat=1, lon=0).
    # Formula: x' = x cos - y sin, y' = x sin + y cos.
    # Here x=lon, y=lat.
    # lon=0, lat=1.
    # lon' = 0*cos - 1*sin = -1
    # lat' = 0*sin + 1*cos = 0
    # Wait, the function signature is rotate_coordinates(lat, lon, angle).
    # Returns lat_rot, lon_rot.
    r_lat, r_lon = feature_ops.rotate_coordinates(
        np.array([1.0]), np.array([0.0]), angle_degrees=90
    )
    print(
        f"    Rotation (lat=1, lon=0, 90deg) -> lat': {r_lat[0]:.4f}, lon': {r_lon[0]:.4f}"
    )

    # cos(90)=0, sin(90)=1
    # lon_rot = 0*0 - 1*1 = -1
    # lat_rot = 0*1 + 1*0 = 0
    # So expected lat_rot=0, lon_rot=-1
    assert np.isclose(r_lat[0], 0.0, atol=1e-7), "Rotation Latitude incorrect"
    assert np.isclose(r_lon[0], -1.0, atol=1e-7), "Rotation Longitude incorrect"

    print(">>> Feature Operations Verified.\n")


def run_pipeline():
    """
    Runs the DataHandler and EnsembleLearner pipeline.
    """
    print(">>> Running Pipeline...")

    # ---------------------------------------------------------
    # 1. Data Handling
    # ---------------------------------------------------------
    print("--- Step 1: Data Handling ---")
    dh = DataHandler()

    # Load and process
    # We force load_cached_data=False to ensure processing logic runs
    train_df, val_df, test_df = dh.load_and_process_data(load_cached_data=False)

    # Validation
    print(f"    Processed Train Shape: {train_df.shape}")
    print(f"    Processed Val Shape:   {val_df.shape}")
    print(f"    Processed Test Shape:  {test_df.shape}")

    # Check for generated features
    expected_cols = [
        "dist_haversine",
        "dist_manhattan",
        "hour",
        "weekday",
        "pickup_latitude_rot",
    ]
    for col in expected_cols:
        assert col in train_df.columns, f"Missing feature: {col}"

    # Check for target
    assert "fare_amount" in train_df.columns, "Target 'fare_amount' missing from train"
    assert "fare_amount" in val_df.columns, "Target 'fare_amount' missing from val"

    # Create subsets
    subsets = dh.create_subsets(train_df)
    assert (
        len(subsets) == config.ENSEMBLE_CONFIG["n_models"]
    ), "Incorrect number of subsets created"

    # ---------------------------------------------------------
    # 2. Ensemble Training
    # ---------------------------------------------------------
    print("\n--- Step 2: Ensemble Training ---")
    learner = EnsembleLearner()

    # Train
    learner.train_ensemble_loop(subsets, val_df)

    # Check if models were saved
    for i in range(config.ENSEMBLE_CONFIG["n_models"]):
        model_path = os.path.join(config.WORKING_DIR, f"model_{i}.json")
        assert os.path.exists(model_path), f"Model file {model_path} was not created."

    # ---------------------------------------------------------
    # 3. Prediction & Submission
    # ---------------------------------------------------------
    print("\n--- Step 3: Prediction & Submission ---")

    # Predict
    preds = learner.predict_ensemble(test_df)

    assert len(preds) == len(test_df), "Prediction length mismatch"
    assert not np.isnan(preds).any(), "Predictions contain NaNs"
    assert (
        preds >= config.CLEANING_PARAMS["min_fare_floor"]
    ).all(), "Predictions violate min fare floor"

    # Generate Submission
    sub_path = config.DATA_PATHS["submission"]
    learner.generate_submission(test_df, sub_path)

    assert os.path.exists(sub_path), "Submission file not found"

    # Verify Submission Content
    sub_df = pd.read_csv(sub_path)
    print(f"    Submission Shape: {sub_df.shape}")
    assert list(sub_df.columns) == [
        "key",
        "fare_amount",
    ], "Submission columns incorrect"
    assert len(sub_df) == len(test_df), "Submission row count mismatch"

    print(">>> Pipeline execution successful.")


if __name__ == "__main__":
    # Ensure reproducibility
    np.random.seed(42)

    try:
        setup_demo_environment()
        test_feature_ops()
        run_pipeline()
        print("\nAll demonstrations completed successfully.")
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        raise e
