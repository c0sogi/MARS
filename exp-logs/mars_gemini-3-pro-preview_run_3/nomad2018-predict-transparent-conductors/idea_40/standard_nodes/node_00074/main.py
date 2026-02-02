import os
import sys
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Import from provided libraries
from library.config import TARGET_COLS, SUBMISSION_PATH
from library.preprocessing import get_preprocessed_data
from library.model import XGBoostRegressorWrapper, generate_submission


def main():
    print("Starting pipeline execution...")

    # 1. Load and Preprocess Data
    # We load cached data if available to save time, otherwise it computes from scratch
    print("Loading Training Data...")
    train_df = get_preprocessed_data("train", load_cached_data=True)

    print("Loading Validation Data...")
    val_df = get_preprocessed_data("val", load_cached_data=True)

    # 2. Configure Model
    model_wrapper = XGBoostRegressorWrapper()

    # Check for GPU availability and configure XGBoost accordingly
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        model_wrapper.params["device"] = "cuda"
        model_wrapper.params["tree_method"] = "hist"
    else:
        print("No GPU detected. Using CPU.")

    # 3. Train Model
    print("Training XGBoost models...")
    model_wrapper.train(train_df, val_df)

    # 4. Validation and Failure Analysis
    print("\n" + "=" * 40)
    print("VALIDATION & FAILURE ANALYSIS")
    print("=" * 40)

    # Prepare validation features (drop non-feature columns)
    drop_cols = TARGET_COLS + ["id", "file_path"]
    X_val = val_df.drop(columns=drop_cols, errors="ignore")

    rmsle_scores = []

    for target in TARGET_COLS:
        print(f"\n--- Analysis for Target: {target} ---")

        # Get true values (already log1p transformed in preprocessing)
        y_true = val_df[target].values

        # Predict using the specific trained model for this target
        # The model predicts in log space
        model = model_wrapper.models[target]
        y_pred = model.predict(X_val)

        # Calculate Metric: RMSE in log space is RMSLE in original space
        mse = mean_squared_error(y_true, y_pred)
        rmsle = np.sqrt(mse)
        rmsle_scores.append(rmsle)
        print(f"RMSLE (Log-Space RMSE): {rmsle:.6f}")

        # Failure Analysis: Correlation between Error Magnitude and Features
        # Calculate absolute error
        errors = np.abs(y_pred - y_true)

        # Create a temporary dataframe for correlation analysis
        analysis_df = X_val.copy()
        analysis_df["__error__"] = errors

        # Compute correlation of all features with the error
        correlations = analysis_df.corr()["__error__"].drop("__error__")

        # Sort by absolute correlation
        top_correlations = correlations.abs().sort_values(ascending=False).head(5)

        print("Top 5 Features correlated with Prediction Error:")
        for feature, corr_val in top_correlations.items():
            # Get the sign from the original correlation series
            sign = correlations[feature]
            print(f"  {feature}: {sign:.4f}")

    # Compute Final Metric (Column-wise RMSLE)
    # Assuming the metric is the mean of the RMSLEs of the columns
    final_metric = np.mean(rmsle_scores)

    print("\n" + "=" * 40)
    print(f"Final Validation Metric: {final_metric}")
    print("=" * 40)

    # 5. Submission Generation
    # Only generate submission if metric is below threshold
    THRESHOLD = 0.05095

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model_wrapper, load_cached_data=True)
    else:
        print(
            f"\nMetric {final_metric} does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
