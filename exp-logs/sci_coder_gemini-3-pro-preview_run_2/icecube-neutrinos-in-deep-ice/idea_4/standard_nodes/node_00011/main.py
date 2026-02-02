import sys
import os
import numpy as np
import pandas as pd
import gc

# =============================================================================
# 1. Configuration & Imports
# =============================================================================
# Override configuration for a fast baseline execution
import library.config

library.config.DEBUG_SAMPLE_SIZE = 300000  # Limit training samples for speed
library.config.N_ESTIMATORS = 1000  # Limit boosting rounds

from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    FEATURE_NAMES,
    SUBMISSION_PATH,
    SEED,
)
from library.data_loader import IceCubeFeatureGenerator
from library.model import GradientBoostingVectorRegressor
from library.inference import generate_submission
from library.utils import setup_logger, spherical_to_cartesian

# Set seeds for reproducibility
np.random.seed(SEED)


def calculate_angular_error(az_true, zen_true, az_pred, zen_pred):
    """
    Calculates the angular error between true and predicted directions.
    Returns the mean angular error and the per-sample errors.
    """
    # Convert to Cartesian vectors
    x_t, y_t, z_t = spherical_to_cartesian(az_true, zen_true)
    x_p, y_p, z_p = spherical_to_cartesian(az_pred, zen_pred)

    # Dot product
    dot_prod = x_t * x_p + y_t * y_p + z_t * z_p

    # Clip to [-1, 1] to avoid numerical errors in arccos
    dot_prod = np.clip(dot_prod, -1.0, 1.0)

    # Angular distance
    errors = np.arccos(dot_prod)

    return np.mean(errors), errors


def main():
    logger = setup_logger("RunFile")
    logger.info("Starting Fast Baseline Pipeline...")

    # =========================================================================
    # 2. Data Preparation
    # =========================================================================
    feature_gen = IceCubeFeatureGenerator()

    logger.info("Loading/Generating Training Data...")
    train_df = feature_gen.process_split(
        meta_path=TRAIN_META_PATH,
        output_path=TRAIN_FEATURES_PATH,
        load_cached_data=True,
    )

    logger.info("Loading/Generating Validation Data...")
    val_df = feature_gen.process_split(
        meta_path=VAL_META_PATH, output_path=VAL_FEATURES_PATH, load_cached_data=True
    )

    # Prepare Feature Matrices
    X_train = train_df[FEATURE_NAMES]
    # Targets are handled internally by the model using 'target_x', 'target_y', 'target_z' columns in train_df

    X_val = val_df[FEATURE_NAMES]

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")

    # =========================================================================
    # 3. Model Training
    # =========================================================================
    model = GradientBoostingVectorRegressor()

    logger.info("Training Model...")
    # We pass the full dataframes as y arguments because the model expects to find target columns inside them
    model.fit(X_train, train_df, X_val, val_df)

    # Free memory
    del train_df, X_train
    gc.collect()

    # =========================================================================
    # 4. Validation & Metric Calculation
    # =========================================================================
    logger.info("Performing Validation Inference...")
    val_preds = model.predict(X_val)

    # Extract Ground Truth
    az_true = val_df["azimuth"].values
    zen_true = val_df["zenith"].values

    az_pred = val_preds["azimuth"].values
    zen_pred = val_preds["zenith"].values

    # Calculate Metric
    mean_error, sample_errors = calculate_angular_error(
        az_true, zen_true, az_pred, zen_pred
    )

    print(f"Final Validation Metric: {mean_error}")

    # =========================================================================
    # 5. Failure Analysis
    # =========================================================================
    logger.info("Performing Failure Analysis...")

    # Create a DataFrame for correlation analysis
    analysis_df = X_val.copy()
    analysis_df["error"] = sample_errors

    # Calculate correlations with error
    correlations = analysis_df.corrwith(analysis_df["error"]).sort_values(
        ascending=False
    )

    print("\nTop Features Correlated with Error:")
    print(correlations.head(5))
    print("\nTop Features Negatively Correlated with Error (Better Performance):")
    print(correlations.tail(5))

    # Clean up validation data
    del val_df, X_val, val_preds, analysis_df
    gc.collect()

    # =========================================================================
    # 6. Submission
    # =========================================================================
    logger.info("Generating Submission...")

    # The inference module handles batching to manage memory
    generate_submission(output_path=SUBMISSION_PATH)

    logger.info("Pipeline Completed Successfully.")


if __name__ == "__main__":
    main()
