import os
import numpy as np
import pandas as pd
import xgboost as xgb
import sys

# Import library modules
import library.config as config
import library.data_loader as data_loader
import library.model_trainer as model_trainer
from library.data_loader import DatasetBuilder


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_small_dataset(source_path, dest_path, n_rows=10000):
    """Creates a small subset of a parquet file for rapid demonstration."""
    print(f"Creating small dataset from {source_path}...")
    # Read a small chunk
    df = pd.read_parquet(source_path).head(n_rows)
    df.to_parquet(dest_path)
    print(f"Saved {len(df)} rows to {dest_path}")
    return dest_path


def main():
    print("Starting Demo Pipeline...")
    set_seed(config.RANDOM_SEED)

    # =========================================================================
    # 1. Setup & Optimization
    # =========================================================================
    # Define paths for small datasets in working directory
    small_train_path = os.path.join(config.WORKING_DIR, "demo_train_small.parquet")
    small_val_path = os.path.join(config.WORKING_DIR, "demo_val_small.parquet")

    # Create small subsets to ensure the demo runs in < 1 minute
    # We use the existing metadata files as source
    create_small_dataset(config.TRAIN_DATA_PATH, small_train_path, n_rows=15000)
    create_small_dataset(config.VAL_DATA_PATH, small_val_path, n_rows=5000)

    # PATCHING: Override constants in data_loader to use our small datasets
    # This is necessary because the module loads constants at import time
    print("Patching data_loader paths for speed optimization...")
    data_loader.TRAIN_DATA_PATH = small_train_path
    data_loader.VAL_DATA_PATH = small_val_path

    # Reduce subsample size to match our small dataset
    # (Original is 5,000,000, we set to 10,000 for demo)
    data_loader.SUBSAMPLE_SIZE = 10000

    # =========================================================================
    # 2. Data Processing (Stage 1 & 2)
    # =========================================================================
    print("\n=== Initializing DatasetBuilder ===")
    builder = DatasetBuilder()

    # Process Training Data
    # load_cached_data=False forces the re-computation to demonstrate the logic
    print("Generating Training Data...")
    train_df = builder.get_train_data(load_cached_data=False)

    # Validation: Check structure
    print(f"Train Data Shape: {train_df.shape}")
    expected_cols = ["base_margin", "distance_haversine", "fare_amount"]
    for col in expected_cols:
        assert col in train_df.columns, f"Missing column {col} in training data"

    # Verify Base Margin Logic (Vectorized Subtraction)
    # base_margin should not be null
    assert (
        train_df["base_margin"].isnull().sum() == 0
    ), "Found NaNs in base_margin (train)"

    # Process Validation Data
    print("Generating Validation Data...")
    val_df = builder.get_val_data(load_cached_data=False)
    assert val_df["base_margin"].isnull().sum() == 0, "Found NaNs in base_margin (val)"

    # Process Test Data
    print("Generating Test Data...")
    test_df = builder.get_test_data(load_cached_data=False)
    assert "fare_amount" not in test_df.columns, "Target leaked into test set"
    assert "base_margin" in test_df.columns, "base_margin missing from test set"

    # =========================================================================
    # 3. Model Training
    # =========================================================================
    print("\n=== Training Model ===")

    # Define fast hyperparameters for demonstration
    demo_params = config.XGB_PARAMS.copy()
    demo_params.update(
        {
            "n_estimators": 10,  # Very few trees for speed
            "max_depth": 4,  # Shallow trees
            "learning_rate": 0.1,
            "n_jobs": 4,
            "tree_method": "hist",  # Use hist (CPU/GPU) for small data compat
        }
    )

    model = model_trainer.train_model(train_df, val_df, params=demo_params)

    # Validate model object
    assert model.model is not None, "XGBoost model was not instantiated correctly"
    assert model.best_iteration is not None, "Best iteration not recorded"
    print(f"Model trained. Best iteration: {model.best_iteration}")

    # =========================================================================
    # 4. Prediction & Submission
    # =========================================================================
    print("\n=== Generating Submission ===")

    # Define output path
    submission_path = os.path.join(config.SUBMISSION_DIR, "demo_submission.csv")

    model_trainer.generate_submission(model, test_df, submission_path=submission_path)

    # Verify Submission
    assert os.path.exists(submission_path), "Submission file not created"

    sub_df = pd.read_csv(submission_path)
    print(f"Submission Shape: {sub_df.shape}")
    print(sub_df.head())

    # Check constraints
    # 1. Row count matches test set
    assert len(sub_df) == len(
        test_df
    ), f"Submission row count mismatch. Expected {len(test_df)}, got {len(sub_df)}"
    # 2. Required columns
    assert (
        "key" in sub_df.columns and "fare_amount" in sub_df.columns
    ), "Incorrect submission columns"
    # 3. No negative fares (enforced by model_trainer, but verifying)
    assert (
        sub_df["fare_amount"] >= config.MIN_FARE
    ).all(), f"Found fares below minimum {config.MIN_FARE}"
    # 4. No NaNs
    assert sub_df["fare_amount"].isnull().sum() == 0, "Submission contains NaNs"

    print("\nDemo Pipeline Completed Successfully.")


if __name__ == "__main__":
    main()
