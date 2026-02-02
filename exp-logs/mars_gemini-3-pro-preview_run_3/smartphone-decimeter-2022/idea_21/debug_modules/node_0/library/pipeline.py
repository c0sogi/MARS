import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_CACHE_PATH,
    VAL_CACHE_PATH,
    TEST_CACHE_PATH,
    WORKING_DIR,
)
from library.feature_eng import create_dataset
from library.model import LGBMResidualPredictor
from library.optimizer import generate_submission

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)


def _load_or_create_dataset(metadata_path, cache_path, load_cached_data=True):
    """
    Helper function to load a dataset from parquet cache or create it from scratch
    using the feature engineering module.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_path (str): Path where the processed parquet file should be stored.
        load_cached_data (bool): If True, attempt to load from cache first.

    Returns:
        pd.DataFrame: The processed dataset with features and targets.
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached dataset from {cache_path}...")
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Creating dataset from metadata: {metadata_path}...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata_df = pd.read_csv(metadata_path)

    # Generate features (this handles per-drive caching internally)
    df = create_dataset(metadata_df, load_cached_data=load_cached_data)

    # 3. Save to cache
    try:
        print(f"Saving dataset to cache: {cache_path}...")
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save dataset cache: {e}")

    return df


def run_training_pipeline(load_cached_data=True):
    """
    Executes the training pipeline:
    1. Loads/Generates Training and Validation datasets.
    2. Trains the LightGBM Residual Predictor using GroupKFold.
    3. Evaluates the model on the hold-out validation set.

    Args:
        load_cached_data (bool): Whether to use cached datasets.

    Returns:
        LGBMResidualPredictor: The trained model instance.
    """
    print("--- Starting Training Pipeline ---")

    # Load Training Data
    train_df = _load_or_create_dataset(
        TRAIN_METADATA_PATH, TRAIN_CACHE_PATH, load_cached_data
    )
    print(f"Training Data Shape: {train_df.shape}")

    # Load Validation Data
    val_df = _load_or_create_dataset(
        VAL_METADATA_PATH, VAL_CACHE_PATH, load_cached_data
    )
    print(f"Validation Data Shape: {val_df.shape}")

    # Initialize and Train Model
    predictor = LGBMResidualPredictor()
    predictor.train(train_df, val_df)

    print("--- Training Pipeline Completed ---")
    return predictor


def run_inference_pipeline(model, load_cached_data=True):
    """
    Executes the inference pipeline:
    1. Loads/Generates Test dataset (features).
    2. Predicts ENU residuals using the trained model.
    3. Runs Global Graph Optimization to generate the final submission.

    Args:
        model (LGBMResidualPredictor): Trained model instance.
        load_cached_data (bool): Whether to use cached datasets.
    """
    print("--- Starting Inference Pipeline ---")

    # Load Test Data
    test_df = _load_or_create_dataset(
        TEST_METADATA_PATH, TEST_CACHE_PATH, load_cached_data
    )
    print(f"Test Data Shape: {test_df.shape}")

    # Generate ML Predictions (Anchors)
    print("Predicting residuals for test set...")
    ml_predictions = model.predict(test_df)

    # Load Test Metadata for the optimizer (needs drive_id/phone_name mapping)
    test_metadata_df = pd.read_csv(TEST_METADATA_PATH)

    # Run Optimization and Generate Submission
    print("Running Global Graph Optimization...")
    generate_submission(
        test_metadata_df, ml_predictions, load_cached_data=load_cached_data
    )

    print("--- Inference Pipeline Completed ---")
