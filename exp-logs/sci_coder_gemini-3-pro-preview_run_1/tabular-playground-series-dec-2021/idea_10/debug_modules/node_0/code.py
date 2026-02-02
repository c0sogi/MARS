import os
import pandas as pd
import numpy as np
import shutil
import logging

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, setup_logger, calculate_accuracy
from library.data_manager import load_data
from library.encoders import MultiClassTargetEncoder
from library.training_engine import run_cv_pipeline


def create_temp_datasets(n_samples=1000):
    """
    Creates small subsets of the original data for demonstration purposes.
    """
    print(f"Creating temporary datasets with {n_samples} samples...")

    # Read original metadata
    # Note: We use the metadata files as source since they are guaranteed to exist per instructions
    orig_train_path = "./metadata/train.csv"
    orig_test_path = "./metadata/test.csv"

    if not os.path.exists(orig_train_path) or not os.path.exists(orig_test_path):
        raise FileNotFoundError(
            "Metadata files not found. Ensure ./metadata/train.csv exists."
        )

    # Load and sample
    df_train = pd.read_csv(orig_train_path, nrows=n_samples)
    df_test = pd.read_csv(orig_test_path, nrows=n_samples)

    # Define temp paths
    temp_train_path = "./working/demo_train.csv"
    temp_test_path = "./working/demo_test.csv"

    # Save temp files
    df_train.to_csv(temp_train_path, index=False)
    df_test.to_csv(temp_test_path, index=False)

    return temp_train_path, temp_test_path


def demonstrate_utils():
    """
    Demonstrates and verifies utility functions.
    """
    print("\n--- Demonstrating Utils ---")

    # 1. Test Seeding
    set_seed(42)
    r1 = np.random.rand()
    set_seed(42)
    r2 = np.random.rand()
    assert r1 == r2, "Random seed did not produce reproducible results."
    print("Seeding verification passed.")

    # 2. Test Metric Calculation
    y_true = np.array([1, 2, 3, 1])
    y_pred = np.array([1, 2, 1, 1])  # 3 correct, 1 wrong
    acc = calculate_accuracy(y_true, y_pred)
    assert acc == 0.75, f"Accuracy calculation failed. Expected 0.75, got {acc}"
    print(f"Accuracy calculation verified: {acc}")


def demonstrate_encoder():
    """
    Demonstrates and verifies the MultiClassTargetEncoder.
    """
    print("\n--- Demonstrating MultiClassTargetEncoder ---")

    # Create synthetic data
    # Feature 'cat' has values 'A' and 'B'.
    # Target classes are 0 and 1.
    df = pd.DataFrame({"cat": ["A", "A", "B", "B", "A"], "target": [0, 1, 0, 0, 0]})

    # Class 0 priors: 4/5 = 0.8
    # Class 1 priors: 1/5 = 0.2

    # For 'A': Total=3, Class0=2, Class1=1
    # For 'B': Total=2, Class0=2, Class1=0

    encoder = MultiClassTargetEncoder(columns=["cat"], smoothing=1.0)
    encoder.fit(df[["cat"]], df["target"])

    transformed = encoder.transform(df[["cat"]])

    # Check if new columns exist
    expected_cols = ["cat_target_0", "cat_target_1"]
    for col in expected_cols:
        assert col in transformed.columns, f"Missing encoded column: {col}"

    # Check values are probabilities (0 to 1)
    assert transformed[expected_cols].min().min() >= 0
    assert transformed[expected_cols].max().max() <= 1

    print("Encoder fit and transform successful.")
    print("Transformed Data Head:\n", transformed.head())


def run_demo_pipeline():
    """
    Runs the full training pipeline using the modified configuration.
    """
    print("\n--- Running Full Pipeline Demo ---")

    # Execute the pipeline provided in library/training_engine.py
    # This will use the monkey-patched Config settings
    run_cv_pipeline()

    # Verify Submission
    submission_path = Config.SUBMISSION_FILE
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not generated at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission generated with shape: {df_sub.shape}")

    # Check format
    expected_cols = ["Id", "Cover_Type"]
    if not all(col in df_sub.columns for col in expected_cols):
        raise ValueError(
            f"Submission missing required columns. Found: {df_sub.columns}"
        )

    # Check row count matches our temp test set (1000)
    assert len(df_sub) == 1000, f"Expected 1000 predictions, found {len(df_sub)}"

    print("Pipeline execution and submission verification passed.")


if __name__ == "__main__":
    # Initialize Logger
    logger = setup_logger("demo_script", log_file="./working/demo.log")

    # --- 1. Setup & Configuration Override ---
    # We modify the Config class attributes directly to adapt the library for a quick demo run.

    # Paths
    temp_train, temp_test = create_temp_datasets(n_samples=1000)
    Config.TRAIN_CSV = temp_train
    Config.TEST_CSV = temp_test
    Config.CACHE_DIR = "./working/demo_cache"
    Config.SUBMISSION_FILE = "./working/demo_submission.csv"

    # Training Hyperparameters (Optimized for speed)
    Config.N_FOLDS = 2
    Config.XGB_PARAMS["n_estimators"] = 10  # Very few trees for speed
    Config.XGB_PARAMS["early_stopping_rounds"] = 5
    Config.XGB_PARAMS["device"] = "cpu"  # Avoid GPU init overhead for tiny data
    Config.XGB_PARAMS["tree_method"] = "hist"

    # Ensure cache directory is clean to force reprocessing of new temp data
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # --- 2. Run Demonstrations ---
    try:
        demonstrate_utils()
        demonstrate_encoder()

        # Verify Data Loading separately before full pipeline
        print("\n--- Verifying Data Manager ---")
        train_df, test_df = load_data(load_cached_data=False)
        assert (
            "Euclidean_Distance_To_Hydrology" in train_df.columns
        ), "Geometric features missing."
        assert (
            "Wilderness_Area_Index" in train_df.columns
        ), "Dense categorical features missing."
        print(f"Data Loaded. Train: {train_df.shape}, Test: {test_df.shape}")

        # Run Full Pipeline
        run_demo_pipeline()

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        raise e
