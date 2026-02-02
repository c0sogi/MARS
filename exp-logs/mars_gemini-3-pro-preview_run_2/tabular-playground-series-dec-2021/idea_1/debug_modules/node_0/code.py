import os
import sys
import numpy as np
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.utils import setup_logger, save_submission
from library.data_manager import load_dataset, LabelMapper
from library.model_trainer import GradientBoostingTrainer


def run_demo():
    # --- 1. Setup & Configuration Overrides (Optimize for Speed) ---
    # We modify the Config class directly to ensure the demo runs quickly
    # and uses a small subset of data.

    print("--- Setting up Configuration for Fast Demonstration ---")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 2000  # Small sample size for quick execution
    Config.NUM_BOOST_ROUND = 10  # Minimal rounds to test training loop
    Config.EARLY_STOPPING_ROUNDS = 5
    Config.VERBOSE_EVAL = -1  # Suppress evaluation logging

    # Ensure LightGBM is silent
    Config.MODEL_PARAMS["verbose"] = -1
    Config.MODEL_PARAMS["n_jobs"] = 4  # Limit threads for demo

    # Set up logger
    logger = setup_logger("demo_script")
    logger.info("Configuration updated for demo mode.")

    # --- 2. Data Loading & Validation ---
    logger.info("--- Demonstrating Data Loading ---")

    # Force reload from metadata to demonstrate processing logic
    X_train, y_train, X_val, y_val, X_test, test_ids = load_dataset(
        load_cached_data=False, debug=Config.DEBUG
    )

    # Validate Data Shapes
    logger.info(f"Train data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")

    if len(X_train) != Config.DEBUG_SAMPLE_SIZE:
        raise AssertionError(
            f"Expected {Config.DEBUG_SAMPLE_SIZE} training rows, got {len(X_train)}"
        )

    if len(y_train) != Config.DEBUG_SAMPLE_SIZE:
        raise AssertionError("Mismatch between X_train and y_train lengths.")

    # Validate Target Encoding (0-indexed)
    unique_targets = np.unique(y_train)
    if unique_targets.min() < 0 or unique_targets.max() >= Config.NUM_CLASSES:
        raise AssertionError(
            f"Target values out of bounds [0, {Config.NUM_CLASSES-1}]: {unique_targets}"
        )

    logger.info("Data loading and shape validation passed.")

    # --- 3. Label Mapper Logic Verification ---
    logger.info("--- Verifying LabelMapper Logic ---")

    # Create dummy original labels based on known classes (e.g., 1, 2, 7)
    dummy_original = pd.Series([1, 2, 7, 1])

    # Encode
    dummy_encoded = LabelMapper.encode(dummy_original)
    expected_encoded = np.array([0, 1, 5, 0])  # Based on Config.TARGET_MAPPING

    if not np.array_equal(dummy_encoded, expected_encoded):
        raise AssertionError(
            f"LabelMapper.encode failed. Got {dummy_encoded}, expected {expected_encoded}"
        )

    # Decode
    dummy_decoded = LabelMapper.decode(dummy_encoded)

    if not np.array_equal(dummy_decoded, dummy_original.values):
        raise AssertionError(
            f"LabelMapper.decode failed. Got {dummy_decoded}, expected {dummy_original.values}"
        )

    logger.info("LabelMapper logic verified.")

    # --- 4. Model Training ---
    logger.info("--- Demonstrating Model Training ---")

    trainer = GradientBoostingTrainer()

    # Train the model
    trainer.train(X_train, y_train, X_val, y_val)

    # Verify model artifact creation
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file was not created at {Config.MODEL_SAVE_PATH}"
        )

    logger.info("Model training completed and artifact verified.")

    # --- 5. Inference & Submission ---
    logger.info("--- Demonstrating Inference and Submission ---")

    # Predict (returns 0-indexed class indices)
    y_pred_indices = trainer.predict(X_test)

    if len(y_pred_indices) != len(X_test):
        raise AssertionError("Prediction length mismatch.")

    # Decode back to original Cover_Type labels
    y_pred_original = LabelMapper.decode(y_pred_indices)

    # Verify decoded values are valid original classes
    valid_classes = set(Config.TARGET_MAPPING.keys())
    if not set(np.unique(y_pred_original)).issubset(valid_classes):
        raise AssertionError("Predictions contain invalid class labels.")

    # Save submission
    save_submission(
        ids=test_ids, predictions=y_pred_original, output_path=Config.SUBMISSION_PATH
    )

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file not found.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    if list(df_sub.columns) != ["Id", "Cover_Type"]:
        raise AssertionError(f"Incorrect submission columns: {df_sub.columns}")

    if len(df_sub) != Config.DEBUG_SAMPLE_SIZE:
        raise AssertionError(
            f"Submission row count mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(df_sub)}"
        )

    logger.info(f"Submission successfully generated at {Config.SUBMISSION_PATH}")
    logger.info("Demo completed successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    np.random.seed(42)

    try:
        run_demo()
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        # Re-raise to ensure the run is marked as failed if something goes wrong
        raise e
