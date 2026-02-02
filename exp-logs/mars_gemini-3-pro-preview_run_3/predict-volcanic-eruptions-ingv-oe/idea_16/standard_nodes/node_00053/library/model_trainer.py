import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error

from library.config import (
    LGBM_PARAMS,
    N_ESTIMATORS,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    SEED,
    SUBMISSION_PATH,
)
from library.utils import seed_everything


def train_lgbm_fold(X_train, y_train, X_val, y_val, params=None):
    """
    Trains a single LightGBM model for a specific fold with early stopping.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        params (dict, optional): LightGBM parameters. Defaults to LGBM_PARAMS.

    Returns:
        lgb.Booster: The trained LightGBM model.
    """
    if params is None:
        params = LGBM_PARAMS.copy()

    # Create LightGBM datasets
    # Reference allows LightGBM to align bins between train and val
    train_ds = lgb.Dataset(X_train, label=y_train)
    val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

    # Define callbacks for early stopping and logging
    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
        lgb.log_evaluation(period=VERBOSE_EVAL),
    ]

    # Train the model
    # Note: verbosity is controlled via params and callbacks
    model = lgb.train(
        params=params,
        train_set=train_ds,
        num_boost_round=N_ESTIMATORS,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    return model


def run_stratified_cv(X, y, n_folds=5, test_X=None):
    """
    Performs Stratified K-Fold Cross-Validation.

    Args:
        X (pd.DataFrame): Feature matrix (index should be segment_id).
        y (pd.Series): Target variable.
        n_folds (int): Number of cross-validation folds.
        test_X (pd.DataFrame, optional): Test feature matrix for generating predictions.

    Returns:
        tuple: (oof_preds, test_preds_avg, models)
            - oof_preds (pd.Series): Out-of-fold predictions indexed by segment_id.
            - test_preds_avg (np.ndarray or None): Averaged predictions for the test set.
            - models (list): List of trained lgb.Booster objects.
    """
    # Ensure reproducibility
    seed_everything(SEED)

    # Create bins for stratification since the target is continuous
    # We use quantiles to ensure balanced bins
    num_bins = 15
    y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    # Initialize arrays for storing predictions
    oof_preds = np.zeros(len(X))
    test_preds_accum = np.zeros(len(test_X)) if test_X is not None else None

    models = []

    print(f"Starting Stratified K-Fold CV with {n_folds} folds...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
        print(f"\n--- Fold {fold + 1} / {n_folds} ---")

        # Split data
        # iloc is used because split returns positional indices
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Train model for this fold
        model = train_lgbm_fold(X_train, y_train, X_val, y_val)
        models.append(model)

        # Generate OOF predictions
        # num_iteration=model.best_iteration uses the best iteration found by early stopping
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred

        # Calculate and print fold metric
        fold_mae = mean_absolute_error(y_val, val_pred)
        print(f"Fold {fold + 1} MAE: {fold_mae}")

        # Generate Test predictions if test set is provided
        if test_X is not None:
            test_pred = model.predict(test_X, num_iteration=model.best_iteration)
            test_preds_accum += test_pred

    # Calculate overall OOF score
    overall_mae = mean_absolute_error(y, oof_preds)
    print(f"\nOverall CV MAE: {overall_mae}")

    # Average the test predictions across folds
    test_preds_avg = None
    if test_preds_accum is not None:
        test_preds_avg = test_preds_accum / n_folds

    # Convert OOF predictions to Series with original index (segment_id)
    oof_series = pd.Series(oof_preds, index=X.index, name="oof_predictions")

    return oof_series, test_preds_avg, models


def generate_submission(test_ids, predictions):
    """
    Saves the test predictions to the submission file.

    Args:
        test_ids (array-like): The segment_ids for the test set.
        predictions (array-like): The predicted time_to_eruption values.
    """
    submission_df = pd.DataFrame(
        {"segment_id": test_ids, "time_to_eruption": predictions}
    )

    # Ensure the directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
