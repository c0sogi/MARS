import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Import from the provided library
from library.config import TARGET_COLS, RANDOM_SEED
from library.data import (
    load_datasets,
    preprocess_targets,
    prepare_matrices,
    inverse_transform_targets,
)
from library.model import train_target_model, make_predictions


def main():
    # Set random seeds
    np.random.seed(RANDOM_SEED)

    print("Starting runfile execution...")

    # 1. Load Data
    # This triggers the feature extraction pipeline (cached if available)
    df_train, df_val, df_test = load_datasets(load_cached_data=True)

    # 2. Preprocess Targets
    # Apply log1p transformation: z = log(1 + y)
    # This is crucial because the metric is RMSLE. Minimizing MSE on z is equivalent to minimizing MSLE on y.
    df_train_log = preprocess_targets(df_train)
    df_val_log = preprocess_targets(df_val)

    # 3. Prepare Feature Matrices
    # This selects relevant numeric features, drops constants, and handles NaNs
    X_train, y_train_log, X_val, y_val_log, X_test, feature_names = prepare_matrices(
        df_train_log, df_val_log, df_test
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")

    # 4. Train Models & Validate
    models = {}
    val_preds_log = {}
    metrics = {}

    for target in TARGET_COLS:
        # Train
        model = train_target_model(
            X_train,
            y_train_log[target],
            X_val,
            y_val_log[target],
            target_name=target,
            early_stopping_rounds=50,  # Aggressive early stopping for speed in this run
        )
        models[target] = model

        # Predict on Validation (Log space)
        preds_log = model.predict(X_val)
        val_preds_log[target] = preds_log

        # Calculate RMSLE for this column
        # Since we are in log space, RMSE here IS the RMSLE of the original data
        # RMSLE = sqrt(mean( (log(1+y) - log(1+y_pred))^2 ))
        # Here y_val_log is log(1+y) and preds_log is log(1+y_pred)
        mse = mean_squared_error(y_val_log[target], preds_log)
        rmsle = np.sqrt(mse)
        metrics[target] = rmsle
        print(f"Target: {target} | Val RMSLE: {rmsle:.6f}")

    # 5. Compute Final Metric
    # Column-wise root mean squared logarithmic error is the mean of the RMSLEs of the columns
    avg_rmsle = np.mean(list(metrics.values()))
    print(f"Final Validation Metric: {avg_rmsle}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # We analyze correlation between absolute error and features to find sources of error
    error_correlations = {}

    for target in TARGET_COLS:
        # Calculate absolute error in log space
        abs_error = np.abs(y_val_log[target] - val_preds_log[target])

        # Create a temporary dataframe with errors and features
        analysis_df = X_val.copy()
        analysis_df["abs_error"] = abs_error

        # Compute correlation
        corr = analysis_df.corr()["abs_error"].drop("abs_error")

        # Get top correlated features with error (magnitude)
        top_corr = corr.abs().sort_values(ascending=False).head(5)
        print(f"\nTop features correlated with error for {target}:")
        print(top_corr)

    # 7. Submission Generation
    # Threshold check as per requirements
    THRESHOLD = 0.05671048180307451

    if avg_rmsle < THRESHOLD:
        print(
            f"\nValidation metric {avg_rmsle} is below threshold {THRESHOLD}. Generating submission..."
        )

        submission_data = {"id": df_test["id"].values}

        for target in TARGET_COLS:
            # Predict in log space
            preds_log = make_predictions(models[target], X_test)

            # Inverse transform to original space
            preds_orig = inverse_transform_targets(preds_log)

            submission_data[target] = preds_orig

        submission_df = pd.DataFrame(submission_data)

        # Ensure output directory exists
        os.makedirs("submission", exist_ok=True)
        submission_path = "submission/submission.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nValidation metric {avg_rmsle} is NOT below threshold {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
