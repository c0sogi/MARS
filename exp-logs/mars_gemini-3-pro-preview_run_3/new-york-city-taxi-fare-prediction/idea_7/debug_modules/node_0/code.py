import os
import shutil
import numpy as np
import pandas as pd
import warnings
import joblib
from datetime import datetime

# Import library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.feature_engineering as fe_module
import library.model_definitions as model_defs
import library.ensemble_trainer as trainer_module
import library.inference_pipeline as inference_module


def main():
    # 1. Setup and Configuration
    print("Setting up demonstration environment...")
    warnings.filterwarnings("ignore")
    # Set fixed seeds for reproducibility
    np.random.seed(42)
    os.environ["PYTHONHASHSEED"] = "42"

    # Define temporary directories for the demo to ensure isolation
    DEMO_BASE = "./demo_env"
    DEMO_INPUT = os.path.join(DEMO_BASE, "input")
    DEMO_WORKING = os.path.join(DEMO_BASE, "working")
    DEMO_SUBMISSION = os.path.join(DEMO_BASE, "submission")

    # Clean up previous runs if any
    if os.path.exists(DEMO_BASE):
        shutil.rmtree(DEMO_BASE)
    os.makedirs(DEMO_INPUT)
    os.makedirs(DEMO_WORKING)
    os.makedirs(DEMO_SUBMISSION)

    # 2. Create Mock Data
    print("Creating dummy datasets for rapid execution...")

    # Generate synthetic data within valid bounding boxes (NYC area)
    # Bounding Box Ref: Lat 40-42, Lon -75 to -73
    n_rows = 500

    data = {
        "key": [f"id_{i}" for i in range(n_rows)],
        "fare_amount": np.random.uniform(5, 50, n_rows),
        "pickup_datetime": [datetime(2015, 1, 1, 12, 0, 0) for _ in range(n_rows)],
        "pickup_longitude": np.random.uniform(-74.05, -73.95, n_rows),
        "pickup_latitude": np.random.uniform(40.70, 40.80, n_rows),
        "dropoff_longitude": np.random.uniform(-74.05, -73.95, n_rows),
        "dropoff_latitude": np.random.uniform(40.70, 40.80, n_rows),
        "passenger_count": np.random.randint(1, 5, n_rows),
    }

    df_train = pd.DataFrame(data)
    # Use same data for validation for simplicity
    df_val = pd.DataFrame(data)
    # Test set (excludes target)
    df_test = df_train.drop(columns=["fare_amount"])

    # Save mock data to temporary input directory
    train_path = os.path.join(DEMO_INPUT, "train.parquet")
    val_path = os.path.join(DEMO_INPUT, "val.parquet")
    test_path = os.path.join(DEMO_INPUT, "test.parquet")

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)

    submission_path = os.path.join(DEMO_SUBMISSION, "submission.csv")

    # 3. Patch Library Modules at Runtime
    # We redirect file paths to our mock data and reduce model complexity for speed.
    print("Patching library configurations...")

    # Patch Data Paths in data_loader
    data_loader.TRAIN_PATH = train_path
    data_loader.VAL_PATH = val_path
    data_loader.TEST_PATH = test_path
    data_loader.WORKING_DIR = DEMO_WORKING

    # Patch Working Directory in other modules
    fe_module.WORKING_DIR = DEMO_WORKING
    trainer_module.WORKING_DIR = DEMO_WORKING
    trainer_module.SUBMISSION_PATH = submission_path
    inference_module.WORKING_DIR = DEMO_WORKING
    inference_module.TEST_PATH = test_path
    inference_module.SUBMISSION_PATH = submission_path

    # Patch Model Hyperparameters (Reduce n_estimators from 5000 to 10)
    # This ensures the training finishes in seconds.
    model_defs.XGB_PARAMS["n_estimators"] = 10
    model_defs.LGBM_PARAMS["n_estimators"] = 10

    # 4. Demonstrate Utils
    print("\n=== Demonstrating Library Utils ===")

    # Test Haversine Distance
    # Distance between (0,0) and (0,1) degrees is approx 111km
    dist = utils.haversine_distance(0, 0, 0, 1)
    print(f"Haversine Distance (0,0) to (0,1): {dist:.4f} km")
    assert 100 < dist < 120, "Haversine calculation is outside expected range."

    # Test Coordinate Rotation
    # Rotating (1,0) by 45 degrees (approx 0.785 rad)
    # x=0 (lon), y=1 (lat). x' = -sin(theta), y' = cos(theta)
    lat_rot, lon_rot = utils.rotate_coordinates(
        np.array([1.0]), np.array([0.0]), np.radians(45)
    )
    print(f"Rotation of (1,0) by 45 deg: Lat={lat_rot[0]:.4f}, Lon={lon_rot[0]:.4f}")
    assert np.isclose(lat_rot[0], 0.7071, atol=1e-3), "Rotation logic incorrect."

    # Test Memory Reduction
    df_mem = pd.DataFrame({"a": np.ones(100, dtype=np.float64)})
    df_reduced = utils.reduce_mem_usage(df_mem)
    print(f"Memory Reduction: {np.float64} -> {df_reduced['a'].dtype}")
    assert (
        df_reduced["a"].dtype == np.float32
    ), "Memory reduction failed to downcast float."

    # 5. Demonstrate Ensemble Training
    print("\n=== Demonstrating Ensemble Trainer ===")

    trainer = trainer_module.EnsembleTrainer()

    # Execute the training pipeline
    # load_cached_data=False forces the loader to read our patched (mock) parquet files
    # debug=True enables sampling (though our data is already small, this verifies the logic)
    print("Starting training stack...")
    results = trainer.train_stack(load_cached_data=False, debug=True, debug_size=200)

    print("Training complete.")
    print(f"Validation RMSE: {results['rmse']:.4f}")

    # Verify that models were saved to the working directory
    # The trainer appends the suffix (e.g., '_debug') to filenames
    assert os.path.exists(
        os.path.join(DEMO_WORKING, "xgb_model_debug.joblib")
    ), "XGB model file missing."
    assert os.path.exists(
        os.path.join(DEMO_WORKING, "lgbm_model_debug.joblib")
    ), "LGBM model file missing."
    assert os.path.exists(
        os.path.join(DEMO_WORKING, "meta_model_debug.joblib")
    ), "Meta model file missing."

    # 6. Demonstrate Inference Pipeline
    print("\n=== Demonstrating Inference Pipeline ===")

    # Generate submission using the models trained above
    # We use model_suffix="_debug" to load the specific models we just trained
    submission_df = inference_module.generate_submission(
        load_cached_data=True,  # Should pick up the cached test features from the training step
        model_suffix="_debug",
        debug=False,
    )

    print("Inference complete.")
    print(f"Submission shape: {submission_df.shape}")

    # Verify submission file integrity
    assert os.path.exists(submission_path), "Submission CSV file missing."

    loaded_sub = pd.read_csv(submission_path)
    assert list(loaded_sub.columns) == [
        "key",
        "fare_amount",
    ], "Submission columns are incorrect."
    assert len(loaded_sub) == n_rows, "Submission row count does not match test set."
    assert (
        not loaded_sub["fare_amount"].isnull().any()
    ), "Submission contains NaN values."

    print("\n=== Demonstration Successful ===")

    # Cleanup temporary environment
    shutil.rmtree(DEMO_BASE)


if __name__ == "__main__":
    main()
