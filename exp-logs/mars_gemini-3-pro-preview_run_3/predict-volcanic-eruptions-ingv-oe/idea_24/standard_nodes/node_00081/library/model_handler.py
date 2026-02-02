import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from library.config import get_lgbm_params, N_FOLDS, SEED, SUBMISSION_PATH, WORKING_DIR
from library.data_manager import get_train_data, get_test_data


def train_fold_model(X_train, y_train, X_val, y_val, params):
    """
    Trains a single LightGBM model for a specific fold.

    Args:
        X_train, y_train: Training data for the fold.
        X_val, y_val: Validation data for the fold.
        params (dict): LightGBM hyperparameters.

    Returns:
        lgb.LGBMRegressor: The trained model.
    """
    # Extract early stopping rounds if present in params
    # We remove it from params to pass it explicitly to the callback
    es_rounds = params.pop("early_stopping_rounds", 100)

    # Initialize model
    model = lgb.LGBMRegressor(**params)

    # Configure callbacks
    # early_stopping: Stops training if metric doesn't improve
    # log_evaluation: Suppress per-iteration logging (verbose=0)
    callbacks = [
        lgb.early_stopping(stopping_rounds=es_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    # Train
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=callbacks,
    )

    return model


def train_ensemble(load_cached_data=True, debug_size=None):
    """
    Trains an ensemble of LightGBM models using Stratified K-Fold Cross Validation.
    Uses ONLY the training set to ensure the validation set remains a valid hold-out.

    Args:
        load_cached_data (bool): Whether to load features from cache.
        debug_size (int): Limit data size for debugging.

    Returns:
        list: A list of trained lgb.LGBMRegressor models.
    """
    print("Loading training data...")
    X, y = get_train_data(load_cached_data=load_cached_data, debug_size=debug_size)

    print(f"Total training samples: {len(X)}")

    # Prepare Stratified K-Fold
    # We must bin the continuous target 'time_to_eruption' to use StratifiedKFold
    # Ensure we have enough bins, but not more than samples/folds allows
    num_bins = min(20, len(y) // N_FOLDS)
    # Fallback for extremely small debug sizes
    if num_bins < 2:
        num_bins = 2

    y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    models = []
    oof_preds = np.zeros(len(y))

    # Get base parameters
    base_params = get_lgbm_params()

    print(f"Starting {N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
        X_fold_train = X.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_val = y.iloc[val_idx]

        # Train model (pass a copy of params because we pop keys inside)
        model = train_fold_model(
            X_fold_train, y_fold_train, X_fold_val, y_fold_val, base_params.copy()
        )

        # Evaluate on fold validation set
        val_preds = model.predict(X_fold_val)
        oof_preds[val_idx] = val_preds

        fold_mae = np.mean(np.abs(y_fold_val - val_preds))
        print(f"Fold {fold + 1}/{N_FOLDS} MAE: {fold_mae:.6f}")

        models.append(model)

    overall_mae = np.mean(np.abs(y - oof_preds))
    print(f"Overall CV MAE: {overall_mae:.6f}")

    return models


def predict_ensemble(models, X_test):
    """
    Generates predictions by averaging outputs from all ensemble models.

    Args:
        models (list): List of trained models.
        X_test (pd.DataFrame): Test features.

    Returns:
        np.array: Averaged predictions.
    """
    if not models:
        raise ValueError("No models provided for prediction.")

    predictions = np.zeros(len(X_test))

    for model in models:
        predictions += model.predict(X_test)

    predictions /= len(models)
    return predictions


def generate_submission(load_cached_data=True, debug_size=None):
    """
    End-to-end pipeline: Train ensemble, predict on test set, save submission.

    Args:
        load_cached_data (bool): Whether to use cached features.
        debug_size (int): Limit data size for debugging.
    """
    # 1. Train Ensemble
    models = train_ensemble(load_cached_data=load_cached_data, debug_size=debug_size)

    # 2. Load Test Data
    print("Loading test data...")
    X_test, segment_ids = get_test_data(
        load_cached_data=load_cached_data, debug_size=debug_size
    )

    # 3. Predict
    print("Generating predictions...")
    predictions = predict_ensemble(models, X_test)

    # 4. Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"segment_id": segment_ids, "time_to_eruption": predictions}
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # 5. Save
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print("Sample predictions:")
    print(submission_df.head())
