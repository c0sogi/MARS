import os
import sys
import numpy as np
import pandas as pd
import warnings
import logging

# Import from the provided library
from library.config import SEED
from library.data_loader import generate_dataset
from library.model_handler import VolcanoLGBM
from library.trainer import run_cv, generate_final_submission
from library.utils import setup_logger

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Setup Logger
logger = setup_logger(name="demo_script", level=logging.INFO)


def demo_data_loading_and_feature_engineering():
    """
    Demonstrates how to load raw sensor data and extract features using the
    generate_dataset function. Uses a small debug_size for speed.
    """
    logger.info("--- 1. Demonstrating Data Loading & Feature Engineering ---")

    # Define a small debug size to process only a few files
    DEBUG_SIZE = 20

    # 1. Generate Training Data
    # This will load metadata, read CSVs, extract wavelet/spectral features, and return X, y
    logger.info(f"Generating training data (debug_size={DEBUG_SIZE})...")
    X_train, y_train = generate_dataset(
        mode="train",
        load_cached_data=False,  # Force processing from scratch
        debug_size=DEBUG_SIZE,
        n_jobs=2,  # Reduce parallelism for this small demo
    )

    # Validation
    assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
    assert isinstance(y_train, pd.Series), "y_train should be a Series"
    assert (
        len(X_train) == DEBUG_SIZE
    ), f"Expected {DEBUG_SIZE} training samples, got {len(X_train)}"
    assert (
        len(y_train) == DEBUG_SIZE
    ), f"Expected {DEBUG_SIZE} targets, got {len(y_train)}"
    assert (
        "segment_id" not in X_train.columns
    ), "segment_id should be dropped from features"

    logger.info(f"Successfully generated Train Data: Shape {X_train.shape}")

    # 2. Generate Test Data
    # For test, y is None, and X includes segment_id (though generate_dataset returns df, None)
    # Note: generate_dataset for 'test' returns (df, None). df contains segment_id.
    logger.info(f"Generating test data (debug_size={DEBUG_SIZE})...")
    df_test, _ = generate_dataset(
        mode="test", load_cached_data=False, debug_size=DEBUG_SIZE, n_jobs=2
    )

    # Validation
    assert isinstance(df_test, pd.DataFrame), "Test data should be a DataFrame"
    assert len(df_test) == DEBUG_SIZE, f"Expected {DEBUG_SIZE} test samples"
    assert "segment_id" in df_test.columns, "Test dataframe must contain segment_id"

    logger.info(f"Successfully generated Test Data: Shape {df_test.shape}")

    return X_train, y_train


def demo_single_model_training(X, y):
    """
    Demonstrates how to instantiate and train the VolcanoLGBM model wrapper.
    """
    logger.info("\n--- 2. Demonstrating Single Model Training ---")

    # Define fast hyperparameters for demonstration
    fast_params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": -1,
        "n_estimators": 20,  # Very low for speed
        "n_jobs": 1,
        "verbosity": -1,
        "seed": SEED,
        "device": "cpu",
    }

    # Split data manually for this specific demo step
    split_idx = int(len(X) * 0.8)
    X_tr, y_tr = X.iloc[:split_idx], y.iloc[:split_idx]
    X_va, y_va = X.iloc[split_idx:], y.iloc[split_idx:]

    logger.info("Initializing VolcanoLGBM...")
    model = VolcanoLGBM(params=fast_params)

    logger.info("Training model...")
    model.train(X_tr, y_tr, X_va, y_va, verbose_eval=10)

    # Test Prediction
    preds = model.predict(X_va)

    # Validation
    assert len(preds) == len(X_va), "Prediction length mismatch"
    assert not np.isnan(preds).any(), "Predictions contain NaNs"

    logger.info(
        f"Training successful. Validation MAE on subset: {np.mean(np.abs(y_va - preds)):.4f}"
    )


def demo_cross_validation_workflow():
    """
    Demonstrates the full Cross-Validation workflow using library.trainer.run_cv.
    """
    logger.info("\n--- 3. Demonstrating Cross-Validation Workflow ---")

    # Fast parameters
    fast_params = {
        "objective": "regression",
        "metric": "mae",
        "n_estimators": 20,
        "num_leaves": 15,
        "learning_rate": 0.1,
        "verbosity": -1,
        "seed": SEED,
        "n_jobs": 1,
    }

    # Run CV with 2 folds and small debug size
    # Note: run_cv internally loads train and val metadata, combines them, and splits.
    # We use debug_size=50 to ensure we have enough data for 2 folds (stratified binning needs samples).
    DEBUG_SIZE_CV = 50
    N_SPLITS = 2

    logger.info(f"Running {N_SPLITS}-fold CV with debug_size={DEBUG_SIZE_CV}...")

    models, overall_mae = run_cv(
        n_splits=N_SPLITS,
        load_cached_data=False,  # Force re-creation to ensure clean demo
        debug_size=DEBUG_SIZE_CV,
        params=fast_params,
    )

    # Validation
    assert len(models) == N_SPLITS, f"Expected {N_SPLITS} trained models"
    assert overall_mae > 0, "MAE should be positive"

    logger.info(f"CV Completed. Overall OOF MAE: {overall_mae:.4f}")
    return models


def demo_submission_generation(models):
    """
    Demonstrates generating the final submission file using the trained models.
    """
    logger.info("\n--- 4. Demonstrating Submission Generation ---")

    output_path = "./working/demo_submission.csv"
    DEBUG_SIZE_TEST = 20

    logger.info(f"Generating submission for {DEBUG_SIZE_TEST} test samples...")

    generate_final_submission(
        models=models,
        load_cached_data=False,
        debug_size=DEBUG_SIZE_TEST,
        output_path=output_path,
    )

    # Validation
    assert os.path.exists(output_path), "Submission file was not created"

    sub_df = pd.read_csv(output_path)
    assert list(sub_df.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "Incorrect submission columns"
    assert (
        len(sub_df) == DEBUG_SIZE_TEST
    ), f"Expected {DEBUG_SIZE_TEST} rows in submission"
    assert not sub_df.isnull().values.any(), "Submission contains Null values"

    logger.info(f"Submission successfully saved to {output_path}")
    logger.info("Head of submission:")
    print(sub_df.head())


if __name__ == "__main__":
    # Ensure reproducibility
    np.random.seed(SEED)

    try:
        # Step 1: Data Loading
        X_sample, y_sample = demo_data_loading_and_feature_engineering()

        # Step 2: Single Model Training
        demo_single_model_training(X_sample, y_sample)

        # Step 3: Cross-Validation
        trained_models = demo_cross_validation_workflow()

        # Step 4: Submission
        demo_submission_generation(trained_models)

        logger.info("\n=== All Demonstrations Completed Successfully ===")

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise e
