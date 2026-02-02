import os
import numpy as np
import pandas as pd
import xgboost as xgb

# Import functions and configurations from the provided library files
from library.config import XGB_PARAMS
from library.feature_engineering import get_target_encoded_data
from library.model_training import train_model, evaluate_model, predict_and_submit


def main():
    # Set fixed seed for reproducibility
    np.random.seed(42)

    print("=== Starting Runfile Execution ===")

    # ---------------------------------------------------------
    # 1. Configuration for Fast Baseline
    # ---------------------------------------------------------
    # Limit training samples to ensure execution completes quickly
    TRAIN_SAMPLE_SIZE = 2_000_000

    # Override default XGB parameters for speed and GPU usage
    # Reducing n_estimators and ensuring GPU histogram method
    fast_xgb_params = XGB_PARAMS.copy()
    fast_xgb_params.update(
        {
            "n_estimators": 5000,
            "learning_rate": 0.05,
            "tree_method": "hist",
            "device": "cuda",
            "n_jobs": 12,
        }
    )

    # ---------------------------------------------------------
    # 2. Data Loading & Preparation
    # ---------------------------------------------------------
    print("Loading data...")

    # Load Training Data (Target Encoded)
    # Uses caching mechanism in library to avoid re-computing if possible
    train_df = get_target_encoded_data("train", load_cached_data=True)

    # Sample training data
    if len(train_df) > TRAIN_SAMPLE_SIZE:
        print(
            f"Sampling training set from {len(train_df)} to {TRAIN_SAMPLE_SIZE} rows."
        )
        train_df_sampled = train_df.sample(n=TRAIN_SAMPLE_SIZE, random_state=42)
    else:
        train_df_sampled = train_df

    # Load Validation Data
    # Must use full validation set for the final metric
    val_df = get_target_encoded_data("val", load_cached_data=True)

    # ---------------------------------------------------------
    # 3. Model Training
    # ---------------------------------------------------------
    print("Training XGBoost model...")
    model, features = train_model(train_df_sampled, val_df, params=fast_xgb_params)

    # ---------------------------------------------------------
    # 4. Evaluation
    # ---------------------------------------------------------
    print("Evaluating model on full validation set...")
    rmse = evaluate_model(model, val_df, features)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {rmse}")

    # ---------------------------------------------------------
    # 5. Failure Analysis
    # ---------------------------------------------------------
    print("\n=== Failure Analysis ===")
    # Create DMatrix for efficient prediction
    dval = xgb.DMatrix(val_df[features])
    preds = model.predict(dval)

    # Calculate residuals
    actuals = val_df["fare_amount"].values
    errors = np.abs(preds - actuals)

    # Analyze correlations
    # We create a temporary dataframe with features and error
    analysis_df = val_df[features].copy()
    analysis_df["error_magnitude"] = errors

    # Compute correlation of features with error magnitude
    correlations = analysis_df.corrwith(analysis_df["error_magnitude"])
    correlations = correlations.drop("error_magnitude").sort_values(
        ascending=False, key=abs
    )

    print("Top 10 Features correlated with Prediction Error:")
    print(correlations.head(10))

    # ---------------------------------------------------------
    # 6. Submission
    # ---------------------------------------------------------
    THRESHOLD = 4.278504866347902

    if rmse < THRESHOLD:
        print(
            f"\nValidation RMSE ({rmse}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        # Load Test Data
        test_df = get_target_encoded_data("test", load_cached_data=True)

        # Generate and Save Submission
        predict_and_submit(model, test_df, features)
    else:
        print(
            f"\nValidation RMSE ({rmse}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )

    print("=== Execution Complete ===")


if __name__ == "__main__":
    main()
