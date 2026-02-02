import pandas as pd
import numpy as np
import xgboost as xgb
from library.config import XGB_PARAMS, EARLY_STOPPING_ROUNDS, TARGET_COLS
from library.utils import log_transform, inverse_log_transform, calculate_rmsle


def train_model(train_df, val_df):
    """
    Trains separate XGBoost regressors for each target variable using the provided training and validation data.

    Args:
        train_df (pd.DataFrame): DataFrame containing training features and targets.
        val_df (pd.DataFrame): DataFrame containing validation features and targets.

    Returns:
        dict: A dictionary mapping target names to trained XGBoost models.
    """
    # Define columns to exclude from features
    # 'id' and 'file_path' are metadata
    # TARGET_COLS are the labels
    exclude_cols = ["id", "file_path"] + TARGET_COLS

    # Identify feature columns
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    print(f"Training with {len(feature_cols)} features.")

    # Prepare feature matrices
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]

    models = {}
    val_predictions = {}

    for target in TARGET_COLS:
        print(f"\nTraining model for target: {target}")

        # Prepare targets with log transformation
        # We transform targets to handle the wide range of values and ensure positivity
        y_train_log = log_transform(train_df[target].values)
        y_val_log = log_transform(val_df[target].values)

        # Initialize XGBoost Regressor with parameters from config
        # Cite debug_lesson_1: Update XGBoost Early Stopping Syntax for Versions 1.6+
        model = xgb.XGBRegressor(
            **XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS
        )

        # Fit the model
        # Using early stopping to prevent overfitting based on validation performance
        model.fit(
            X_train,
            y_train_log,
            eval_set=[(X_val, y_val_log)],
            verbose=False,
        )

        # Generate predictions on validation set (still in log scale)
        y_pred_log = model.predict(X_val)

        # Inverse transform predictions to original scale
        y_pred = inverse_log_transform(y_pred_log)

        # Store predictions for overall metric calculation
        val_predictions[target] = y_pred

        # Calculate and print RMSLE for this specific target
        y_val_true = val_df[target].values
        rmsle = calculate_rmsle(y_val_true, y_pred)
        print(f"Validation RMSLE for {target}: {rmsle}")

        # Store the trained model
        models[target] = model

    # Calculate Overall RMSLE across all targets
    # Create DataFrames to ensure alignment of columns
    y_true_all = val_df[TARGET_COLS].values
    y_pred_all = pd.DataFrame(val_predictions)[TARGET_COLS].values

    overall_rmsle = calculate_rmsle(y_true_all, y_pred_all)
    print(f"\nOverall Validation RMSLE: {overall_rmsle}")

    return models


def predict_model(models, test_df):
    """
    Generates predictions for the test set using the trained models.

    Args:
        models (dict): Dictionary of trained models keyed by target name.
        test_df (pd.DataFrame): DataFrame containing test features.

    Returns:
        pd.DataFrame: Submission DataFrame containing 'id' and predicted targets.
    """
    # Identify feature columns (must match training features)
    exclude_cols = ["id", "file_path"] + TARGET_COLS
    feature_cols = [c for c in test_df.columns if c not in exclude_cols]

    X_test = test_df[feature_cols]

    # Initialize submission dictionary with IDs
    submission_data = {"id": test_df["id"]}

    for target in TARGET_COLS:
        if target not in models:
            raise ValueError(
                f"Model for target '{target}' not found in the provided models dictionary."
            )

        model = models[target]

        # Generate predictions in log scale
        y_pred_log = model.predict(X_test)

        # Inverse transform to original scale
        y_pred = inverse_log_transform(y_pred_log)

        submission_data[target] = y_pred

    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Ensure column order matches sample submission
    cols_order = ["id"] + TARGET_COLS
    submission_df = submission_df[cols_order]

    return submission_df
