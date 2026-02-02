import os
import gc
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

# Import from provided libraries
from library.config import (
    SEED,
    TRAIN_SUBSAMPLE_SIZE,
    NUM_BOOST_ROUND,
    EARLY_STOPPING_ROUNDS,
    SUBMISSION_PATH,
    XGB_PARAMS,
)
from library.data_processor import get_processed_data
from library.feature_engineering import InteractionStatsEngine
from library.model_trainer import XGBoostTrainer


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    set_seed(SEED)

    # Define constants for this run
    # We use the config defaults but ensure explicit control here
    SUBSAMPLE = TRAIN_SUBSAMPLE_SIZE
    ROUNDS = NUM_BOOST_ROUND
    # Threshold from task description
    RMSE_THRESHOLD = 3.5069767944123895

    print("Starting End-to-End Pipeline...")

    # =========================================================================
    # Phase 1: Wisdom (Global Stats Generation)
    # =========================================================================
    print("\n=== Phase 1: Wisdom (Global Stats Generation) ===")
    # Load strict data for robust stats
    df_strict = get_processed_data("train", mode="strict", load_cached_data=True)

    # Initialize and Fit Engine
    # This computes mean/sum/count for geohash routes on the full dataset
    engine = InteractionStatsEngine()
    engine.fit(df_strict, load_cached=True)

    del df_strict
    gc.collect()

    # =========================================================================
    # Phase 2: Learner (Training Data Prep)
    # =========================================================================
    print("\n=== Phase 2: Learner (Training Data Prep) ===")
    # Load training subsample
    df_train = get_processed_data(
        "train", mode="loose", subsample_size=SUBSAMPLE, load_cached_data=True
    )

    # Transform using Vectorized Subtraction (prevents leakage)
    df_train = engine.transform_train(df_train)

    # Define Feature Columns
    exclude_cols = {"key", "fare_amount", "pickup_datetime", "fold_id"}
    # Exclude raw geohash strings
    exclude_cols.update(
        [
            c
            for c in df_train.columns
            if "geohash" in c and "mean" not in c and "pickup" in c and "dropoff" in c
        ]
    )
    # Actually, the engine output keeps raw geohash columns (e.g. pickup_geohash_7).
    # We must exclude them from features.
    exclude_cols.update(
        [c for c in df_train.columns if isinstance(df_train[c].iloc[0], str)]
    )

    features = [c for c in df_train.columns if c not in exclude_cols]
    target = "fare_amount"

    print(f"Features ({len(features)}): {features}")

    X_train = df_train[features]
    y_train = df_train[target]

    # Keep df_train in memory? No, we have X_train, y_train.
    del df_train
    gc.collect()

    # =========================================================================
    # Phase 3: Validation Data Prep
    # =========================================================================
    print("\n=== Phase 3: Validation Data Prep ===")
    df_val = get_processed_data("val", mode="loose", load_cached_data=True)

    # Transform using Direct Mapping (Validation is unseen)
    df_val = engine.transform_test(df_val)

    X_val = df_val[features]
    y_val = df_val[target]

    # We keep X_val, y_val for failure analysis later
    # df_val can be deleted if we don't need other columns, but failure analysis might want metadata?
    # The requirement says "correlation between the model's error magnitude and the input features".
    # X_val has the input features.
    del df_val
    gc.collect()

    # =========================================================================
    # Phase 4: Model Training
    # =========================================================================
    print("\n=== Phase 4: Model Training ===")
    trainer = XGBoostTrainer(params=XGB_PARAMS)

    trainer.train(
        X_train,
        y_train,
        X_val,
        y_val,
        num_boost_round=ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    # Clean up training data
    del X_train, y_train
    gc.collect()

    # =========================================================================
    # Phase 5: Evaluation & Failure Analysis
    # =========================================================================
    print("\n=== Phase 5: Evaluation & Failure Analysis ===")

    # Generate predictions on validation set
    print("Predicting on Validation Set...")
    val_preds = trainer.predict(X_val)

    # Compute RMSE
    mse = np.mean((y_val - val_preds) ** 2)
    rmse = np.sqrt(mse)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {rmse}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    abs_error = np.abs(y_val - val_preds)

    # Calculate correlation between error magnitude and features
    # We use pandas corrwith for efficiency
    error_series = pd.Series(abs_error, index=X_val.index, name="abs_error")
    correlations = X_val.corrwith(error_series).abs().sort_values(ascending=False)

    print("Top 10 Features correlated with Error Magnitude:")
    print(correlations.head(10))

    # =========================================================================
    # Phase 6: Submission
    # =========================================================================
    if rmse < RMSE_THRESHOLD:
        print(
            f"\nValidation RMSE ({rmse}) is below threshold ({RMSE_THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        df_test = get_processed_data("test", mode="inference", load_cached_data=True)

        # Transform
        df_test = engine.transform_test(df_test)

        X_test = df_test[features]

        # Predict
        test_preds = trainer.predict(X_test)

        # Post-processing: Floor at 2.50
        test_preds = np.maximum(test_preds, 2.50)

        # Create Submission DataFrame
        submission = pd.DataFrame({"key": df_test["key"], "fare_amount": test_preds})

        # Ensure directory exists
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        # Save
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation RMSE ({rmse}) is NOT below threshold ({RMSE_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
