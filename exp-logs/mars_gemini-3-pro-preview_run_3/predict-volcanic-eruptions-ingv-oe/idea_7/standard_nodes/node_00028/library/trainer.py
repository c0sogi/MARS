import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb

from library.dataset import get_train_data, get_test_data
from library.models import get_lightgbm_regressor, get_xgboost_regressor

# Configuration
WORKING_DIR = "./working/idea_optimized"
SUBMISSION_DIR = "./submission"
RANDOM_STATE = 42


def run_stratified_cv(
    n_splits=5,
    debug_size=None,
    load_cached_data=True,
    generate_submission=True,
    n_estimators=10000,
    learning_rate=0.01,
):
    """
    Performs Stratified K-Fold Cross-Validation using LightGBM and XGBoost.
    Trains models, evaluates MAE, and generates a submission file.

    Args:
        n_splits (int): Number of CV folds.
        debug_size (int, optional): Number of samples to use for debugging.
        load_cached_data (bool): Whether to use cached features.
        generate_submission (bool): Whether to generate predictions for the test set.
        n_estimators (int): Max number of boosting rounds.
        learning_rate (float): Learning rate for the models.
    """
    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    if generate_submission:
        os.makedirs(SUBMISSION_DIR, exist_ok=True)

    print("Loading training data...")
    X, y = get_train_data(debug_size=debug_size, load_cached_data=load_cached_data)

    X_test = None
    test_ids = None
    test_preds = None

    if generate_submission:
        print("Loading test data...")
        X_test, test_ids = get_test_data(
            debug_size=debug_size, load_cached_data=load_cached_data
        )
        test_preds = np.zeros(len(X_test))

    # Create stratified bins for continuous target
    # We use Sturges' rule to determine a reasonable number of bins, clamped to a range
    num_bins = int(1 + np.log2(len(y)))
    num_bins = max(2, min(num_bins, 20))  # Clamp between 2 and 20

    try:
        y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")
    except ValueError:
        # Fallback for very small debug datasets where qcut might fail
        y_bins = np.zeros(len(y))

    # Adjust n_splits if dataset is smaller than requested splits (e.g. in debug mode)
    real_n_splits = min(n_splits, len(y))
    if real_n_splits < 2:
        print("Dataset too small for CV. Training on full set without validation.")
        real_n_splits = 1
        # Mock split for loop
        folds = [(np.arange(len(y)), np.arange(len(y)))]
    else:
        skf = StratifiedKFold(
            n_splits=real_n_splits, shuffle=True, random_state=RANDOM_STATE
        )
        folds = list(skf.split(X, y_bins))

    oof_preds = np.zeros(len(y))
    mae_scores = []

    print(f"Starting {real_n_splits}-Fold Stratified CV...")

    for fold, (train_idx, val_idx) in enumerate(folds):
        # In case of n_splits=1 (debug/tiny data), train/val are same, but we proceed for logic consistency
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        # --- Train LightGBM ---
        lgb_model = get_lightgbm_regressor(
            random_state=RANDOM_STATE,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )

        # Configure callbacks for early stopping and logging
        lgb_callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),  # Suppress log printing
        ]

        lgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=lgb_callbacks,
        )

        val_pred_lgb = lgb_model.predict(X_val)

        # --- Train XGBoost ---
        # Using CUDA if available (A100 is present)
        xgb_model = get_xgboost_regressor(
            random_state=RANDOM_STATE,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            device="cuda",
        )

        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_pred_xgb = xgb_model.predict(X_val)

        # --- Ensemble (Average) ---
        val_pred_ens = (val_pred_lgb + val_pred_xgb) / 2.0

        # Store OOF predictions
        oof_preds[val_idx] = val_pred_ens

        # Calculate Fold Metric
        fold_mae = mean_absolute_error(y_val, val_pred_ens)
        mae_scores.append(fold_mae)
        print(f"Fold {fold + 1} MAE: {fold_mae}")

        # --- Inference on Test Set ---
        if generate_submission and X_test is not None:
            pred_test_lgb = lgb_model.predict(X_test)
            pred_test_xgb = xgb_model.predict(X_test)
            pred_test_ens = (pred_test_lgb + pred_test_xgb) / 2.0

            # Accumulate average
            test_preds += pred_test_ens / real_n_splits

    # Overall Evaluation
    total_mae = mean_absolute_error(y, oof_preds)
    print(f"Overall CV MAE: {total_mae}")
    print(f"Average Fold MAE: {np.mean(mae_scores)}")

    # Generate Submission File
    if generate_submission and test_preds is not None:
        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

        sub_df = pd.DataFrame({"segment_id": test_ids, "time_to_eruption": test_preds})

        # Ensure segment_id is integer
        sub_df["segment_id"] = sub_df["segment_id"].astype(int)

        sub_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    return total_mae
