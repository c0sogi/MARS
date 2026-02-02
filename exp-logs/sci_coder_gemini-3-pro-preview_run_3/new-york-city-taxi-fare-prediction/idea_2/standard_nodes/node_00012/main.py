import sys
import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error

# Import library modules
from library.config import Config
from library.utils import set_seed
from library.data_loader import load_and_process_data
from library.engine import train_gbdt, generate_predictions


def main():
    # 1. Setup and Configuration
    set_seed(Config.SEED)

    # 2. Load Data
    print("Loading datasets...")
    # load_cached_data=True will use the parquet files in working/ if they exist
    train_df, val_df, test_df = load_and_process_data(load_cached_data=True)

    # 3. Training
    # Using full dataset as per Lesson solution_lesson_node_00010
    print(f"Training on full dataset: {len(train_df)} rows.")

    # Train GBDT
    print("\n=== Training GBDT Stream ===")
    gbdt_model = train_gbdt(train_df, val_df)

    # 5. Final Validation (Full Set)
    print("\n=== Performing Final Validation on Full Hold-out Set ===")

    # GBDT Predictions
    print("Predicting with GBDT...")
    final_val_preds = gbdt_model.predict(val_df)

    # Metric
    y_val = val_df[Config.TARGET_COL].values
    rmse = np.sqrt(mean_squared_error(y_val, final_val_preds))

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {rmse}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate errors
    errors = y_val - final_val_preds
    abs_errors = np.abs(errors)

    # Create a temporary dataframe for analysis
    analysis_df = val_df.copy()
    analysis_df["abs_error"] = abs_errors

    # Select numerical features for correlation
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target and error cols from features list
    exclude_cols = [Config.TARGET_COL, "abs_error", "prediction", "error"]
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    # Compute correlations
    correlations = (
        analysis_df[feature_cols + ["abs_error"]].corr()["abs_error"].drop("abs_error")
    )
    print("Correlation between Error Magnitude and Features:")
    print(correlations.sort_values(ascending=False).head(10))

    # 7. Submission
    THRESHOLD = 3.3935366001817666

    if rmse < THRESHOLD:
        print(
            f"\nValidation RMSE ({rmse}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        generate_predictions(gbdt_model, test_df)
    else:
        print(
            f"\nValidation RMSE ({rmse}) is NOT below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
