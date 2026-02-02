import os
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error

from library.config import TARGET_COLS, SUBMISSION_FILE, XGB_PARAMS, RANDOM_SEED
from library.data import build_feature_matrix
from library.model import DualTargetRegressor


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def calculate_rmsle(y_true, y_pred):
    """
    Calculate RMSLE.
    Note: Targets are already non-negative.
    """
    # Ensure no negative values for log
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def perform_failure_analysis(val_df, val_preds_df, feature_cols):
    print("\n--- Failure Analysis ---")

    # Extract feature matrix from validation dataframe
    X_val = val_df[feature_cols].select_dtypes(include=[np.number])

    for target in TARGET_COLS:
        y_true = val_df[target]
        y_pred = val_preds_df[target]

        # Calculate absolute error
        error = np.abs(y_true - y_pred)

        # Calculate correlation between features and error
        correlations = X_val.corrwith(error).abs().sort_values(ascending=False)

        print(f"\nTop 5 features correlated with error for {target}:")
        print(correlations.head(5))


def main():
    set_seed(RANDOM_SEED)

    # 1. Setup & Configuration
    device = get_device()
    print(f"Using device: {device}")

    # Update XGBoost params for GPU if available
    xgb_params = XGB_PARAMS.copy()
    if device == "cuda":
        xgb_params["device"] = "cuda"
        # Ensure tree method is compatible
        xgb_params["tree_method"] = "hist"

    # 2. Load Data
    print("Loading datasets...")
    # Using cached data if available to speed up
    train_df = build_feature_matrix("train", load_cached_data=True)
    val_df = build_feature_matrix("val", load_cached_data=True)
    test_df = build_feature_matrix("test", load_cached_data=True)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # 3. Train Model
    print("\nTraining model...")
    model = DualTargetRegressor(xgb_params)
    model.fit(train_df, val_df, verbose=True)

    # 4. Validation & Evaluation
    print("\nEvaluating on validation set...")
    val_preds_df = model.predict(val_df)

    rmsle_scores = []
    for target in TARGET_COLS:
        score = calculate_rmsle(val_df[target], val_preds_df[target])
        rmsle_scores.append(score)
        print(f"{target} RMSLE: {score}")

    # Final Metric: Mean of column-wise RMSLE
    final_metric = np.mean(rmsle_scores)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    # Identify feature columns (exclude targets, id, file_path)
    exclude_cols = ["id", "file_path"] + TARGET_COLS
    feature_cols = [c for c in val_df.columns if c not in exclude_cols]
    perform_failure_analysis(val_df, val_preds_df, feature_cols)

    # 6. Submission
    THRESHOLD = 0.056919346405286564

    if final_metric < THRESHOLD:
        print("\nMetric meets threshold. Generating submission...")
        test_preds_df = model.predict(test_df)

        submission = pd.DataFrame()
        submission["id"] = test_df["id"]
        for target in TARGET_COLS:
            submission[target] = test_preds_df[target].values

        # Ensure output directory exists
        os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)
        submission.to_csv(SUBMISSION_FILE, index=False)
        print(f"Submission saved to {SUBMISSION_FILE}")
    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
