import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

# Import provided library modules
import library.config as config
import library.trainers as trainers
from library.data_pipeline import load_and_process


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Set random seeds for reproducibility
    np.random.seed(config.SEED)

    # Use full dataset for maximum performance (Cite solution_lesson_node_00012)
    SAMPLE_SIZE = None

    print(f"Starting pipeline with Sample Size: {SAMPLE_SIZE} (Full Data)")

    # ==========================================
    # 2. Data Loading & Processing
    # ==========================================
    print("Loading and processing data...")
    # load_and_process handles spatial feature engineering, temporal extraction, and geometric features
    X_train, y_train, X_val, y_val, X_test, test_ids = load_and_process(
        load_cached_data=True, debug_sample_size=SAMPLE_SIZE
    )

    print(f"Data loaded. Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # ==========================================
    # 3. Model Training
    # ==========================================
    # Train XGBoost (Uses GPU: A100)
    print("\n=== Training XGBoost ===")
    xgb_model = trainers.train_xgboost_model(X_train, y_train, X_val, y_val)

    # Train LightGBM (Uses CPU: 12 vCPUs)
    print("\n=== Training LightGBM ===")
    lgbm_model = trainers.train_lightgbm_model(X_train, y_train, X_val, y_val)

    # ==========================================
    # 4. Validation & Ensembling
    # ==========================================
    print("\n=== Validation & Ensembling ===")

    # Generate predictions
    # Note: Models are already in eval mode (sklearn-style API)
    xgb_pred_val = xgb_model.predict(X_val)
    lgbm_pred_val = lgbm_model.predict(X_val)

    # Ensemble (Simple Average)
    ensemble_pred_val = 0.5 * xgb_pred_val + 0.5 * lgbm_pred_val

    # Calculate Metric
    rmse = root_mean_squared_error(y_val, ensemble_pred_val)

    # REQUIRED OUTPUT: Print Final Validation Metric
    print(f"Final Validation Metric: {rmse}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(y_val - ensemble_pred_val)

    # Create analysis dataframe
    analysis_df = X_val.copy()
    analysis_df["abs_error"] = errors

    # Calculate correlations with error
    # We drop columns that might be constant or non-numeric if any slipped through,
    # though the pipeline ensures numeric features.
    correlations = analysis_df.corrwith(analysis_df["abs_error"]).sort_values(
        ascending=False
    )

    print("Top 10 features correlated with prediction error:")
    print(correlations.head(10))

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    THRESHOLD = 3.3935366001817666

    if rmse < THRESHOLD:
        print(
            f"\nValidation RMSE ({rmse}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Test Inference
        xgb_pred_test = xgb_model.predict(X_test)

        # Create Submission DataFrame
        submission = pd.DataFrame({"key": test_ids, "fare_amount": xgb_pred_test})

        # Save
        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation RMSE ({rmse}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
