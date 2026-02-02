import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb

# Import provided library functions
from library.dataset import get_train_data, get_val_data, get_test_data
from library.models import get_lightgbm_regressor, get_xgboost_regressor

# --- Configuration ---
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
THRESHOLD = 2739761.2592384242
RANDOM_STATE = 42
N_SPLITS = 5
# Increased estimators to ensure full convergence (Cite solution_lesson_node_00002)
N_ESTIMATORS = 6000
LEARNING_RATE = 0.02


def main():
    # --- 1. Data Loading ---
    print("Loading datasets...")
    # Load training data (for CV)
    X_train, y_train = get_train_data(load_cached_data=True)
    # Load hold-out validation data (for final metric & failure analysis)
    X_val, y_val = get_val_data(load_cached_data=True)
    # Load test data (for submission)
    X_test, test_ids = get_test_data(load_cached_data=True)

    print(f"Train Set: {X_train.shape}")
    print(f"Val Set: {X_val.shape}")
    print(f"Test Set: {X_test.shape}")

    # --- 2. Device Detection ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for XGBoost: {device}")

    # --- 3. Stratified K-Fold CV ---
    # Binning for stratification to ensure representative folds
    num_bins = int(1 + np.log2(len(y_train)))
    num_bins = max(2, min(num_bins, 20))
    y_bins = pd.qcut(y_train, q=num_bins, labels=False, duplicates="drop")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Accumulators for ensemble predictions
    val_preds_accum = np.zeros(len(y_val))
    test_preds_accum = np.zeros(len(test_ids))

    print(f"Starting {N_SPLITS}-Fold Stratified CV...")

    for fold, (train_idx, fold_val_idx) in enumerate(skf.split(X_train, y_bins)):
        # Prepare Fold Data
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_fol_val, y_fol_val = X_train.iloc[fold_val_idx], y_train[fold_val_idx]

        # --- Model 1: LightGBM ---
        lgb_model = get_lightgbm_regressor(
            random_state=RANDOM_STATE,
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
        )

        # Train LightGBM with Early Stopping
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_fol_val, y_fol_val)],
            eval_metric="mae",
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        # Inference (Accumulate)
        val_preds_accum += lgb_model.predict(X_val)
        test_preds_accum += lgb_model.predict(X_test)

        # --- Model 2: XGBoost ---
        xgb_model = get_xgboost_regressor(
            random_state=RANDOM_STATE,
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            device=device,
        )

        # Train XGBoost
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_fol_val, y_fol_val)], verbose=False)

        # Inference (Accumulate)
        val_preds_accum += xgb_model.predict(X_val)
        test_preds_accum += xgb_model.predict(X_test)

        print(f"Fold {fold + 1} completed.")

    # --- 4. Ensemble Aggregation ---
    # We have summed predictions from (N_SPLITS * 2) models
    n_models = N_SPLITS * 2
    val_preds_final = val_preds_accum / n_models
    test_preds_final = test_preds_accum / n_models

    # --- 5. Validation Assessment ---
    final_mae = mean_absolute_error(y_val, val_preds_final)
    print(f"Final Validation Metric: {final_mae}")

    # --- 6. Failure Analysis ---
    print("\nPerforming Failure Analysis on Hold-out Validation Set...")
    errors = np.abs(y_val - val_preds_final)

    # Calculate correlation between features and error magnitude
    error_series = pd.Series(errors, index=X_val.index)
    correlations = X_val.corrwith(error_series).abs().sort_values(ascending=False)

    print("Top 10 Features correlated with Error Magnitude:")
    print(correlations.head(10))

    # --- 7. Submission Generation ---
    if final_mae < THRESHOLD:
        print(
            f"\nMetric {final_mae} is better than threshold {THRESHOLD}. Generating submission..."
        )
        os.makedirs(SUBMISSION_DIR, exist_ok=True)

        submission_df = pd.DataFrame(
            {"segment_id": test_ids, "time_to_eruption": test_preds_final}
        )

        # Ensure segment_id is integer
        submission_df["segment_id"] = submission_df["segment_id"].astype(int)

        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric {final_mae} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
