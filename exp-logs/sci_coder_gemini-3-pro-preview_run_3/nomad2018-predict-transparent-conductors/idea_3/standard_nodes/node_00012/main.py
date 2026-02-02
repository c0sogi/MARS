import os
import sys
import pandas as pd
import numpy as np
import torch

# Import from provided libraries
from library.config import Config
from library.utils import set_seed, compute_column_wise_rmsle, save_submission
from library.feature_pipeline import (
    build_feature_matrix,
    transform_targets,
    inverse_transform_targets,
)
from library.model import XGBoostRegressorWrapper


def run():
    # 1. Setup
    set_seed(Config.RANDOM_SEED)
    print("Starting runfile.py execution...")

    # 2. Data Loading & Feature Engineering
    # We use the pipeline to get processed features (Metadata + Descriptors + MACE)
    # The pipeline handles caching.

    print("Loading and processing training data...")
    # Loading full training data (1728 samples)
    # The MACE feature extraction utilizes GPU if available, making it efficient.
    train_df = build_feature_matrix(split="train", load_cached_data=True)

    print("Loading and processing validation data...")
    val_df = build_feature_matrix(split="val", load_cached_data=True)

    # Prepare Features and Targets
    # Identify feature columns: drop ID and Targets
    feature_cols = [
        c for c in train_df.columns if c not in Config.TARGET_COLS + [Config.ID_COL]
    ]

    X_train = train_df[feature_cols]
    y_train_raw = train_df[Config.TARGET_COLS]

    X_val = val_df[feature_cols]
    y_val_raw = val_df[Config.TARGET_COLS]

    # Transform targets (Log1p) to align with RMSLE metric
    print("Transforming targets...")
    y_train_log = transform_targets(y_train_raw)
    y_val_log = transform_targets(y_val_raw)

    # 3. Model Training
    print("Initializing and training model...")
    model = XGBoostRegressorWrapper()

    # Fit model
    model.fit(X_train, y_train_log, X_val, y_val_log)

    # 4. Validation Assessment
    print("Evaluating on validation set...")
    # Predict (returns log scale)
    val_preds_log = model.predict(X_val)

    # Inverse transform to original scale
    val_preds = inverse_transform_targets(val_preds_log)

    # Compute Metric
    metric_score, metric_details = compute_column_wise_rmsle(
        y_val_raw, val_preds, Config.TARGET_COLS
    )

    print(f"Final Validation Metric: {metric_score}")
    print("Metric Details:", metric_details)

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute log errors per sample
    # RMSLE is essentially RMSE in log space, so we look at log errors
    log_errors = np.abs(np.log1p(y_val_raw) - np.log1p(val_preds))
    # Average error across targets for a single 'error' metric per sample
    mean_log_error = log_errors.mean(axis=1)

    # Create a dataframe for correlation analysis
    analysis_df = X_val.copy()
    analysis_df["mean_log_error"] = mean_log_error

    # Compute correlations
    correlations = (
        analysis_df.corrwith(analysis_df["mean_log_error"])
        .abs()
        .sort_values(ascending=False)
    )
    print("Top 10 features correlated with error magnitude:")
    print(correlations.head(10))

    # 6. Submission Generation
    threshold = 0.06380692050212411
    if metric_score < threshold:
        print(
            f"\nValidation metric {metric_score} is better than threshold {threshold}. Generating submission..."
        )

        # Load Test Data
        print("Loading and processing test data...")
        test_df = build_feature_matrix(split="test", load_cached_data=True)

        X_test = test_df[feature_cols]
        test_ids = test_df[Config.ID_COL]

        # Predict
        test_preds_log = model.predict(X_test)

        # Inverse Transform
        test_preds = inverse_transform_targets(test_preds_log)

        # Save
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        save_submission(
            test_ids, test_preds.values, Config.TARGET_COLS, submission_path
        )

    else:
        print(
            f"\nValidation metric {metric_score} is NOT better than threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    run()
