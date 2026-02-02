import os
import sys
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import setup_logger
from library.features import FeatureExtractor
from library.dataset import VolcanoDataset
from library.model import ModelTrainer


def run_demo():
    # ==========================================
    # 1. Setup and Configuration Overrides
    # ==========================================
    print("--- 1. Setup and Configuration ---")

    # Set seeds for reproducibility
    np.random.seed(42)

    # Override Config for a fast, isolated demo run
    # We use a specific directory for this demo to keep artifacts separate
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission_demo.csv")

    # Enable Debug mode to process small chunks by default where applicable
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20

    # Reduce Model complexity for instant training
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["verbose"] = -1

    # Ensure directories exist
    Config.setup()

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # ==========================================
    # 2. Demonstrate FeatureExtractor
    # ==========================================
    print("\n--- 2. Demonstrating FeatureExtractor ---")

    extractor = FeatureExtractor()

    # Load raw metadata manually to pass a small slice to the extractor
    train_meta_path = Config.TRAIN_META_PATH
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"Metadata not found at {train_meta_path}")

    train_meta_df = pd.read_csv(train_meta_path)

    # Take a tiny sample (5 rows)
    small_meta = train_meta_df.head(5).copy()

    print(f"Processing {len(small_meta)} segments for feature extraction demo...")

    # Process data (force re-computation by setting load_cached_data=False)
    features_df = extractor.process_data(
        meta_df=small_meta, dataset_name="demo_train_small", load_cached_data=False
    )

    # Validation
    print("Validating Feature Extraction results...")
    assert isinstance(features_df, pd.DataFrame), "Output must be a DataFrame"
    assert len(features_df) == 5, f"Expected 5 rows, got {len(features_df)}"
    assert "segment_id" in features_df.columns, "segment_id column missing"
    assert "time_to_eruption" in features_df.columns, "Target column missing"

    # Check for some expected feature columns (e.g., from kinematics or texture)
    # Based on features.py logic: sensor_1_vel_q50, sensor_1_dwt_energy, etc.
    expected_col_partial = "sensor_1_vel"
    has_feature = any(c.startswith(expected_col_partial) for c in features_df.columns)
    assert (
        has_feature
    ), f"Expected features starting with {expected_col_partial} not found."

    print("Feature Extraction verification passed.")

    # ==========================================
    # 3. Demonstrate VolcanoDataset
    # ==========================================
    print("\n--- 3. Demonstrating VolcanoDataset ---")

    dataset = VolcanoDataset()

    # Retrieve training data with a limit
    # Note: get_train_data handles loading metadata and calling extractor
    limit_rows = 10
    print(f"Loading {limit_rows} training samples via Dataset class...")
    X_train, y_train = dataset.get_train_data(load_cached_data=False, limit=limit_rows)

    # Validation
    assert len(X_train) == limit_rows, f"X_train length mismatch: {len(X_train)}"
    assert len(y_train) == limit_rows, f"y_train length mismatch: {len(y_train)}"
    assert "segment_id" not in X_train.columns, "segment_id should be dropped from X"
    assert "time_to_eruption" not in X_train.columns, "Target should be dropped from X"

    # Retrieve test data
    print(f"Loading {limit_rows} test samples via Dataset class...")
    X_test, test_ids = dataset.get_test_data(load_cached_data=False, limit=limit_rows)

    # Validation
    assert len(X_test) == limit_rows, f"X_test length mismatch: {len(X_test)}"
    assert len(test_ids) == limit_rows, f"test_ids length mismatch: {len(test_ids)}"

    print("Dataset verification passed.")

    # ==========================================
    # 4. Demonstrate ModelTrainer
    # ==========================================
    print("\n--- 4. Demonstrating ModelTrainer ---")

    trainer = ModelTrainer()

    # Train the model
    # We use a slightly larger limit to ensure we have enough data for LightGBM's internal checks
    # though 20 (Config.DEBUG_SAMPLE_SIZE) is usually enough for a dummy run.
    # The trainer uses Config.DEBUG_SAMPLE_SIZE internally if Config.DEBUG is True,
    # or the explicit limit passed to the method.
    train_limit = 50
    print(f"Training model with limit={train_limit}...")

    model = trainer.train(load_cached_data=False, limit=train_limit)

    # Validation
    assert model is not None, "Model object is None after training"
    assert os.path.exists(
        trainer.model_path
    ), f"Model file not created at {trainer.model_path}"

    # Generate Submission
    # Using a small limit for inference
    test_limit = 10
    print(f"Generating submission with limit={test_limit}...")
    trainer.generate_submission(load_cached_data=False, limit=test_limit)

    # Validation
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission Head:")
    print(sub_df.head())

    assert list(sub_df.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "Incorrect submission columns"
    assert (
        len(sub_df) == test_limit
    ), f"Expected {test_limit} predictions, got {len(sub_df)}"
    assert sub_df["segment_id"].dtype == "int64", "segment_id must be int64"

    print("ModelTrainer verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
