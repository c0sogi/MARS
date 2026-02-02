import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import random
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.data_loader import build_dataset
from library.model import (
    train_target_model,
    predict_target,
    generate_submission_file,
    get_feature_columns,
)
from sklearn.metrics import mean_squared_error


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Setup
    set_seed(Config.RANDOM_SEED)
    print("Initializing Physics-Informed Feature Engineering Pipeline...")

    # 2. Load Data & Generate Features
    # load_cached_data=True will use the pre-computed parquet files if they exist
    # This includes geometric, chemical disorder, and electrostatic features
    train_df, val_df, test_df = build_dataset(load_cached_data=True, debug=False)

    # 3. Configure XGBoost for available hardware
    xgb_params = Config.XGB_PARAMS.copy()
    if torch.cuda.is_available():
        print("GPU detected. Configuring XGBoost for CUDA acceleration.")
        xgb_params["device"] = "cuda"
        # 'hist' tree method is required for efficient GPU training
        xgb_params["tree_method"] = "hist"
    else:
        print("No GPU detected. Using CPU.")
        xgb_params["device"] = "cpu"
        xgb_params["tree_method"] = "hist"

    # 4. Train Models and Validate
    targets = Config.TARGET_COLS
    models = {}
    val_scores = []
    val_errors = pd.DataFrame(index=val_df.index)

    # Identify feature columns (excluding metadata and targets)
    feature_cols = get_feature_columns(train_df)
    print(f"Training with {len(feature_cols)} features.")

    for target in targets:
        print(f"\n--- Training model for Target: {target} ---")

        # Train the model with early stopping using the validation set
        model = train_target_model(
            train_df, val_df, target, params=xgb_params, verbose=True
        )
        models[target] = model

        # Generate predictions on validation set (log scale)
        X_val = val_df[feature_cols]
        y_val_log = val_df[target]
        y_pred_log = model.predict(X_val)

        # Calculate RMSLE (which is simply RMSE on the log-transformed data)
        rmsle = np.sqrt(mean_squared_error(y_val_log, y_pred_log))
        val_scores.append(rmsle)
        print(f"Target {target} Validation RMSLE: {rmsle:.6f}")

        # Store absolute errors for failure analysis
        val_errors[f"{target}_error"] = np.abs(y_val_log - y_pred_log)

    # 5. Final Metric Calculation
    # The metric is the column-wise root mean squared logarithmic error
    final_metric = np.mean(val_scores)
    print(f"\nFinal Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis (Correlation with Error Magnitude) ---")
    # Combine feature values with error values to find correlations
    analysis_df = val_df[feature_cols].copy()
    for col in val_errors.columns:
        analysis_df[col] = val_errors[col]

    corr_matrix = analysis_df.corr()

    for target in targets:
        error_col = f"{target}_error"
        if error_col in corr_matrix.columns:
            print(f"\nTop 5 features correlated with high error in {target}:")
            # Filter out the error columns themselves
            error_corrs = corr_matrix[error_col].drop(
                [f"{t}_error" for t in targets], errors="ignore"
            )
            # Sort by absolute correlation
            top_corrs = error_corrs.abs().sort_values(ascending=False).head(5)
            print(top_corrs)

    # 7. Submission Generation
    # Only generate submission if the metric meets the threshold requirement
    THRESHOLD = 0.06278041684313306

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} is lower than threshold {THRESHOLD}.")
        print("Generating predictions for test set...")

        predictions = {}
        for target in targets:
            # predict_target handles feature extraction and inverse transformation (exp(x)-1)
            preds = predict_target(models[target], test_df)
            predictions[target] = preds

        generate_submission_file(test_df, predictions)
    else:
        print(f"\nMetric {final_metric} is NOT lower than threshold {THRESHOLD}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
