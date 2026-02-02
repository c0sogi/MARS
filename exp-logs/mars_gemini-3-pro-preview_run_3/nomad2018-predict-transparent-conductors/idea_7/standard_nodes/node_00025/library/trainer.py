import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.config import Config
from library.feature_pipeline import prepare_features
from library.utils import compute_rmsle, save_submission


def train_xgboost_models(sample_size=None, load_cached_data=True):
    """
    Trains XGBoost models for each target variable using the prepared features.

    Args:
        sample_size (int, optional): Number of samples to use for debugging/testing.
        load_cached_data (bool): Whether to load features from cache if available.

    Returns:
        models (dict): Dictionary of trained XGBoost models keyed by target name.
        feature_columns (list): List of column names used during training (for alignment).
        val_rmsle (float): The validation RMSLE score.
    """
    # 1. Load Training Data
    print("Loading training data...")
    X_train, y_train, _ = prepare_features(
        split="train", sample_size=sample_size, load_cached_data=load_cached_data
    )

    # 2. Load Validation Data
    print("Loading validation data...")
    X_val, y_val, _ = prepare_features(
        split="val", sample_size=sample_size, load_cached_data=load_cached_data
    )

    # 3. Align Columns
    # One-hot encoding of spacegroups might result in different columns for train vs val.
    # We enforce the training columns structure on the validation set.
    feature_columns = X_train.columns.tolist()

    # reindex adds missing columns (filled with 0) and drops extra columns
    X_val = X_val.reindex(columns=feature_columns, fill_value=0)

    # 4. Train Models
    models = {}
    val_preds_log = []
    val_targets_log = []

    print(f"Training XGBoost models with parameters: {Config.XGB_PARAMS}")

    for target in Config.TARGET_COLS:
        print(f"\n--- Training model for target: {target} ---")

        y_train_target = y_train[target]
        y_val_target = y_val[target]

        model = xgb.XGBRegressor(**Config.XGB_PARAMS)

        # Fit model with early stopping using the validation set
        model.fit(
            X_train, y_train_target, eval_set=[(X_val, y_val_target)], verbose=False
        )

        models[target] = model

        # Generate predictions on validation set (still in log space)
        y_pred_val = model.predict(X_val)

        val_preds_log.append(y_pred_val)
        val_targets_log.append(y_val_target.values)

        # Calculate MSE in log space for monitoring
        mse_log = mean_squared_error(y_val_target, y_pred_val)
        print(f"{target} Validation MSE (log-space): {mse_log}")

    # 5. Compute Overall Validation Metric (RMSLE)
    # Stack predictions and targets to shape (n_samples, n_targets)
    val_preds_log = np.column_stack(val_preds_log)
    val_targets_log = np.column_stack(val_targets_log)

    # Inverse transform if log transformation was applied
    if Config.LOG_TRANSFORM_TARGETS:
        val_preds = np.expm1(val_preds_log)
        val_targets = np.expm1(val_targets_log)
    else:
        val_preds = val_preds_log
        val_targets = val_targets_log

    # Clip negative predictions to 0 (energy cannot be negative)
    val_preds = np.maximum(val_preds, 0)

    # Compute metric
    val_rmsle = compute_rmsle(val_targets, val_preds)
    print(f"\nOverall Validation RMSLE: {val_rmsle}")

    return models, feature_columns, val_rmsle


def predict(models, feature_columns, sample_size=None, load_cached_data=True):
    """
    Generates predictions for the test set using the trained models.

    Args:
        models (dict): Dictionary of trained models.
        feature_columns (list): List of feature names to align test data.
        sample_size (int, optional): Number of samples for debugging.
        load_cached_data (bool): Whether to load features from cache.

    Returns:
        ids (pd.Series): Sequence of IDs for the test samples.
        predictions (np.ndarray): 2D array of predictions (n_samples, 2).
    """
    print("\nLoading test data...")
    X_test, _, ids = prepare_features(
        split="test", sample_size=sample_size, load_cached_data=load_cached_data
    )

    # Align test columns to training columns
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    preds_list = []

    for target in Config.TARGET_COLS:
        if target not in models:
            raise ValueError(f"No trained model found for target {target}")

        print(f"Predicting {target}...")
        model = models[target]
        y_pred_log = model.predict(X_test)
        preds_list.append(y_pred_log)

    # Combine predictions
    predictions_log = np.column_stack(preds_list)

    # Inverse transform
    if Config.LOG_TRANSFORM_TARGETS:
        predictions = np.expm1(predictions_log)
    else:
        predictions = predictions_log

    # Clip negative values
    predictions = np.maximum(predictions, 0)

    return ids, predictions


def run_pipeline(sample_size=None, load_cached_data=True):
    """
    Runs the full training and inference pipeline.

    Args:
        sample_size (int, optional): Subset size for debugging.
        load_cached_data (bool): Whether to use cached feature files.

    Returns:
        float: The validation RMSLE score.
    """
    # Train models
    models, feature_cols, val_score = train_xgboost_models(
        sample_size=sample_size, load_cached_data=load_cached_data
    )

    # Generate predictions on test set
    ids, preds = predict(
        models, feature_cols, sample_size=sample_size, load_cached_data=load_cached_data
    )

    # Save submission file
    save_submission(ids, preds, filename="submission.csv")

    return val_score
