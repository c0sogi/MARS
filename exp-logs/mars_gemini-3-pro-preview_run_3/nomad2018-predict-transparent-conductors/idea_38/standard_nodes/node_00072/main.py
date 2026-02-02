import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

# Import from provided libraries
from library.config import RANDOM_SEED, XGB_PARAMS
from library.data_manager import process_dataset
from library.preprocessor import preprocess_features
from library.regressor import DualTargetRegressor, generate_submission


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Setup
    set_seed(RANDOM_SEED)

    # 2. Data Loading & Feature Generation
    # Load metadata and generate features (Anisotropy, BVS, ECoN, RDFs)
    # load_cached_data=True allows using pre-computed parquet files if available
    print("Loading and processing dataset...")
    X_train, y_train, X_val, y_val, X_test, test_ids = process_dataset(
        load_cached_data=True
    )

    # 3. Preprocessing
    # Clean features (e.g., remove constant columns)
    print("Preprocessing features...")
    X_train_clean, X_val_clean, X_test_clean, cleaner = preprocess_features(
        X_train, X_val, X_test, load_cached_data=True
    )

    # 4. Model Configuration
    params = XGB_PARAMS.copy()

    # Configure for GPU if available
    if torch.cuda.is_available():
        print("GPU detected. Configuring XGBoost for GPU acceleration.")
        params["device"] = "cuda"
        params["tree_method"] = "hist"
    else:
        print("No GPU detected. Using CPU.")
        params["n_jobs"] = -1

    # 5. Training
    print("Initializing and training DualTargetRegressor...")
    regressor = DualTargetRegressor(params=params)

    # Train with early stopping to prevent overfitting and speed up convergence
    regressor.fit(
        X_train_clean,
        y_train,
        X_val_clean,
        y_val,
        early_stopping_rounds=100,
        verbose=False,
    )

    # 6. Validation & Evaluation
    print("Evaluating model on validation set...")
    metrics = regressor.evaluate(X_val_clean, y_val)

    # Compute Final Metric: Mean of Column-wise RMSLE
    final_metric = np.mean(list(metrics.values()))
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Generate predictions on validation set (inverse transformed)
    preds_val = regressor.predict(X_val_clean)

    # Calculate error magnitude in log space (since metric is RMSLE)
    # Error = |log(1+true) - log(1+pred)|
    error_per_sample = np.zeros(len(X_val_clean))

    for target in preds_val.columns:
        y_true_log = np.log1p(y_val[target].values)
        y_pred_log = np.log1p(preds_val[target].values)
        error_per_sample += np.abs(y_true_log - y_pred_log)

    # Average error across targets for analysis
    error_per_sample /= len(preds_val.columns)

    # Create analysis dataframe
    analysis_df = X_val_clean.copy()
    analysis_df["error_magnitude"] = error_per_sample

    # Compute correlations with error magnitude
    correlations = analysis_df.corrwith(analysis_df["error_magnitude"])
    correlations = correlations.drop("error_magnitude")

    # Identify top correlated features (positive or negative)
    top_correlations = correlations.abs().sort_values(ascending=False).head(10)

    print("Top 10 features correlated with prediction error:")
    print(top_correlations)

    # 8. Submission
    THRESHOLD = 0.05095
    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric {final_metric} meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(regressor, X_test_clean, test_ids)
    else:
        print(
            f"\nValidation metric {final_metric} does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
