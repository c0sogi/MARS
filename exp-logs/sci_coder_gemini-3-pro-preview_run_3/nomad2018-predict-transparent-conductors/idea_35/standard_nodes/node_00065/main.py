import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Import from library
import library.config as config
from library.preprocessing import prepare_datasets
from library.model import DualTargetRegressor
from library.utils import calculate_rmsle, save_submission


def set_seed(seed=42):
    np.random.seed(seed)
    import random

    random.seed(seed)


def main():
    # 1. Set Random Seeds
    set_seed(config.RANDOM_SEED)

    print("Starting runfile execution...")

    # 2. Load and Prepare Data
    # This handles loading metadata, generating features (cached), cleaning, and log-transforming targets
    print("Preparing datasets...")
    train_df, val_df, test_df = prepare_datasets(load_cached_data=True)

    # 3. Separate Features and Targets
    # Identify feature columns (everything except id and targets)
    feature_cols = [c for c in train_df.columns if c not in config.TARGET_COLS + ["id"]]

    X_train = train_df[feature_cols]
    y_train = train_df[config.TARGET_COLS]

    X_val = val_df[feature_cols]
    y_val = val_df[config.TARGET_COLS]

    X_test = test_df[feature_cols]
    test_ids = test_df["id"]

    print(f"Training features: {len(feature_cols)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # 4. Initialize and Train Model
    # We use the default params from config, which include early stopping
    print("Initializing model...")
    model = DualTargetRegressor()

    print("Training model...")
    # y_train and y_val are already log1p transformed by prepare_datasets
    model.fit(X_train, y_train, X_val, y_val)

    # 5. Validation Assessment
    print("Performing validation inference...")
    # model.predict returns predictions in ORIGINAL scale (expm1 is applied internally)
    val_preds_df = model.predict(X_val)

    # We need ground truth in ORIGINAL scale for RMSLE calculation
    # y_val is log1p transformed, so we apply expm1
    y_val_orig = np.expm1(y_val)

    # Calculate RMSLE
    # Convert DataFrames to numpy arrays for the utility function
    rmsle_score = calculate_rmsle(y_val_orig.values, val_preds_df.values)

    print(f"Final Validation Metric: {rmsle_score}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude per sample
    # We use the difference in log space as a proxy for the error contribution to RMSLE
    # Error = mean( | log(1+y_true) - log(1+y_pred) | ) across targets

    # Re-log transform predictions to compare with y_val (which is log scale)
    val_preds_log = np.log1p(val_preds_df)

    # Absolute error in log space
    errors = np.abs(y_val.values - val_preds_log.values)
    # Mean error across the two targets
    mean_log_error = np.mean(errors, axis=1)

    # Create a dataframe for correlation analysis
    analysis_df = X_val.copy()
    analysis_df["log_error_magnitude"] = mean_log_error

    # Compute correlations
    correlations = (
        analysis_df.corrwith(analysis_df["log_error_magnitude"])
        .abs()
        .sort_values(ascending=False)
    )

    print("Top 10 features correlated with prediction error (log space):")
    print(correlations.drop("log_error_magnitude").head(10))

    # 7. Submission Generation
    THRESHOLD = 0.05095
    if rmsle_score < THRESHOLD:
        print(
            f"\nValidation score ({rmsle_score}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        test_preds_df = model.predict(X_test)

        # Extract values in correct order of TARGET_COLS
        predictions = test_preds_df[config.TARGET_COLS].values

        # Save
        save_submission(test_ids.values, predictions, filename="submission.csv")
    else:
        print(
            f"\nValidation score ({rmsle_score}) does NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
