import os
import sys
import shutil
import numpy as np
import pandas as pd
import warnings

# Ensure library imports work by adding current directory to path
sys.path.append(os.getcwd())

from library.config import Config
from library.data_loader import generate_dataset
from library.model_handler import LGBMRegressorWrapper
from library.signal_utils import compute_moments, impute_nans

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set global seeds for reproducibility
np.random.seed(42)


class DemoConfig(Config):
    """
    Configuration subclass optimized for a quick demonstration run.
    Reduces model complexity and defines a specific working directory.
    """

    # Reduced hyperparameters for speed
    N_ESTIMATORS = 50
    EARLY_STOPPING_ROUNDS = 10
    NUM_LEAVES = 31
    LEARNING_RATE = 0.05

    # Demo-specific paths
    WORKING_DIR = "./working/demo_execution"
    SUBMISSION_DIR = WORKING_DIR
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure silent execution for LightGBM
    VERBOSITY = -1


def verify_signal_utils():
    """
    Validates the correctness of basic signal processing utility functions.
    """
    print("--- Verifying Signal Utils ---")

    # Test NaN imputation
    dummy_signal = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    imputed = impute_nans(dummy_signal)

    # Expected mean is (1+2+4+5)/4 = 3.0
    assert not np.isnan(imputed).any(), "NaN imputation failed: NaNs remain."
    assert np.isclose(
        imputed[2], 3.0
    ), f"Mean imputation incorrect. Expected 3.0, got {imputed[2]}"

    # Test Moment Calculation
    clean_signal = np.array([1, 2, 3, 4, 5])
    moments = compute_moments(clean_signal)

    assert np.isclose(moments["mean"], 3.0), "Mean calculation incorrect."
    assert np.isclose(
        moments["std"], np.std(clean_signal)
    ), "Std calculation incorrect."

    print("Signal Utils verification passed.")


def run_demo_pipeline():
    """
    Executes the full ML pipeline: Data Loading -> Training -> Inference.
    """
    print("\n--- Starting Demo Pipeline ---")

    # 1. Setup Configuration
    cfg = DemoConfig()

    # Clean working directory for a fresh run
    if os.path.exists(cfg.WORKING_DIR):
        shutil.rmtree(cfg.WORKING_DIR)
    os.makedirs(cfg.WORKING_DIR, exist_ok=True)

    # Define sample size for quick execution
    SAMPLE_SIZE = 50

    # 2. Load Training Data
    # We force load_cached_data=False to demonstrate the feature extraction logic
    print(f"Loading Training Data (Subset: {SAMPLE_SIZE} samples)...")
    X_train, y_train, train_ids = generate_dataset(
        metadata_path=cfg.TRAIN_METADATA,
        cfg=cfg,
        load_cached_data=False,
        dataset_name="train",
        sample_size=SAMPLE_SIZE,
    )

    # Validation
    assert (
        len(X_train) == SAMPLE_SIZE
    ), f"Train set size mismatch. Expected {SAMPLE_SIZE}, got {len(X_train)}"
    assert len(y_train) == SAMPLE_SIZE, "Target size mismatch."
    assert not X_train.isnull().values.any(), "Feature matrix contains NaNs."
    print(f"Training data loaded. Shape: {X_train.shape}")

    # 3. Load Validation Data
    print(f"Loading Validation Data (Subset: {SAMPLE_SIZE} samples)...")
    X_val, y_val, val_ids = generate_dataset(
        metadata_path=cfg.VAL_METADATA,
        cfg=cfg,
        load_cached_data=False,
        dataset_name="val",
        sample_size=SAMPLE_SIZE,
    )
    assert len(X_val) == SAMPLE_SIZE, "Validation set size mismatch."
    print(f"Validation data loaded. Shape: {X_val.shape}")

    # 4. Model Training
    print("Initializing and Training Model...")
    model_wrapper = LGBMRegressorWrapper(cfg)

    # Train
    model_wrapper.fit(X_train, y_train, X_val, y_val)

    # Validation
    assert model_wrapper.model is not None, "Model object is None after training."
    print("Model training completed.")

    # 5. Load Test Data
    print(f"Loading Test Data (Subset: {SAMPLE_SIZE} samples)...")
    X_test, _, test_ids = generate_dataset(
        metadata_path=cfg.TEST_METADATA,
        cfg=cfg,
        load_cached_data=False,
        dataset_name="test",
        sample_size=SAMPLE_SIZE,
    )
    assert len(X_test) == SAMPLE_SIZE, "Test set size mismatch."
    print(f"Test data loaded. Shape: {X_test.shape}")

    # 6. Generate Submission
    print("Generating Submission...")
    model_wrapper.generate_submission(X_test, test_ids)

    # Validation
    if not os.path.exists(cfg.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {cfg.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(cfg.SUBMISSION_PATH)

    # Check Header
    expected_cols = ["segment_id", "time_to_eruption"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Invalid columns. Expected {expected_cols}, got {list(sub_df.columns)}"

    # Check Row Count
    assert (
        len(sub_df) == SAMPLE_SIZE
    ), f"Submission row count mismatch. Expected {SAMPLE_SIZE}, got {len(sub_df)}"

    # Check Data Types
    assert pd.api.types.is_numeric_dtype(
        sub_df["segment_id"]
    ), "segment_id should be numeric"
    assert pd.api.types.is_numeric_dtype(
        sub_df["time_to_eruption"]
    ), "time_to_eruption should be numeric"

    print(f"Submission generated successfully at {cfg.SUBMISSION_PATH}")
    print("\n--- Demo Pipeline Completed Successfully ---")


if __name__ == "__main__":
    try:
        verify_signal_utils()
        run_demo_pipeline()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        raise e
