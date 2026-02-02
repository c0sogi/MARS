import os
import pandas as pd
from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    SUBMISSION_PATH,
    FEATURE_NAMES,
)
from library.data_loader import IceCubeFeatureGenerator
from library.model import GradientBoostingVectorRegressor, generate_submission
from library.utils import setup_logger


def run_training(load_cached_data=True):
    """
    Orchestrates the training pipeline:
    1. Generates or loads engineered features for Train/Val/Test splits.
    2. Trains the Gradient Boosting Vector Regressor with Early Stopping.
    3. Generates predictions for the Test set and saves the submission file.

    Args:
        load_cached_data (bool): If True, attempts to load features from Parquet cache.
                                 If False or cache missing, re-computes features.
    """
    logger = setup_logger("TrainingPipeline")
    logger.info("Starting training pipeline...")

    # =========================================================================
    # 1. Feature Engineering & Data Loading
    # =========================================================================
    feature_gen = IceCubeFeatureGenerator()

    # Process Training Data
    logger.info("Preparing Training Data...")
    train_df = feature_gen.process_split(
        meta_path=TRAIN_META_PATH,
        output_path=TRAIN_FEATURES_PATH,
        load_cached_data=load_cached_data,
    )

    # Process Validation Data
    logger.info("Preparing Validation Data...")
    val_df = feature_gen.process_split(
        meta_path=VAL_META_PATH,
        output_path=VAL_FEATURES_PATH,
        load_cached_data=load_cached_data,
    )

    # Filter feature columns for X
    # The dataframes contain targets and metadata, so we select only the engineered features for input
    X_train = train_df[FEATURE_NAMES]
    X_val = val_df[FEATURE_NAMES]

    # y contains the targets (target_x, target_y, target_z) which are present in the full df
    y_train = train_df
    y_val = val_df

    logger.info(f"Training Data Shape: {X_train.shape}")
    logger.info(f"Validation Data Shape: {X_val.shape}")

    # =========================================================================
    # 2. Model Training
    # =========================================================================
    model = GradientBoostingVectorRegressor()

    logger.info("Training GradientBoostingVectorRegressor...")
    metrics = model.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    logger.info("Training completed.")
    for axis, score in metrics.items():
        # Printing full precision as requested
        print(f"Final Validation MSE for component {axis}: {score}")

    # =========================================================================
    # 3. Inference & Submission
    # =========================================================================
    logger.info("Preparing Test Data for Inference...")
    test_df = feature_gen.process_split(
        meta_path=TEST_META_PATH,
        output_path=TEST_FEATURES_PATH,
        load_cached_data=load_cached_data,
    )

    X_test = test_df[FEATURE_NAMES]

    logger.info(f"Generating submission for {len(X_test)} events...")
    generate_submission(model=model, test_features=X_test, output_path=SUBMISSION_PATH)

    logger.info("Pipeline finished successfully.")
