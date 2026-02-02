import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.config import Config
from library.data_loader import inverse_transform_targets


def get_feature_columns(df):
    """
    Identifies feature columns by excluding metadata and target columns.

    Args:
        df (pd.DataFrame): The dataframe containing features and metadata.

    Returns:
        list: List of column names to be used as features.
    """
    # Columns to exclude from training features
    exclude_cols = ["id", "file_path"] + Config.TARGET_COLS

    # Select numeric columns that are not in the exclusion list
    # We explicitly include spacegroup as it is numeric (int) in the provided metadata
    features = [c for c in df.columns if c not in exclude_cols]
    return features


def train_target_model(train_df, val_df, target_col, params=None, verbose=False):
    """
    Trains an XGBoost regressor for a single target variable.

    Args:
        train_df (pd.DataFrame): Training data with features and log-transformed targets.
        val_df (pd.DataFrame): Validation data for early stopping.
        target_col (str): The name of the target column to train on.
        params (dict, optional): XGBoost hyperparameters. Defaults to Config.XGB_PARAMS.
        verbose (bool): Whether to print training progress.

    Returns:
        xgb.XGBRegressor: The trained model.
    """
    if params is None:
        params = Config.XGB_PARAMS.copy()

    feature_cols = get_feature_columns(train_df)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    model = xgb.XGBRegressor(**params)

    if verbose:
        print(f"Training model for target: {target_col}")
        print(f"Input features: {len(feature_cols)}")

    # Train with early stopping to prevent overfitting
    # Note: The metric is RMSE on log-transformed data, which is equivalent to RMSLE on original data
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=100 if verbose else False,
    )

    # Evaluate on validation set
    preds_log = model.predict(X_val)
    rmse_log = np.sqrt(mean_squared_error(y_val, preds_log))

    if verbose:
        print(f"Validation RMSLE for {target_col}: {rmse_log:.6f}")

    return model


def predict_target(model, test_df):
    """
    Generates predictions for a specific target using a trained model.

    Args:
        model (xgb.XGBRegressor): The trained XGBoost model.
        test_df (pd.DataFrame): Test dataset containing features.

    Returns:
        np.array: Predictions in the original scale (inverse transformed).
    """
    feature_cols = get_feature_columns(test_df)
    X_test = test_df[feature_cols]

    # Predict in log space
    log_preds = model.predict(X_test)

    # Convert back to original scale (exp(x) - 1)
    original_scale_preds = inverse_transform_targets(log_preds)

    return original_scale_preds


def generate_submission_file(test_df, predictions_dict, output_path=None):
    """
    Creates the submission CSV file from predictions.

    Args:
        test_df (pd.DataFrame): Test dataframe containing 'id'.
        predictions_dict (dict): Dictionary mapping target column names to prediction arrays.
        output_path (str, optional): Path to save the CSV. Defaults to Config.SUBMISSION_PATH.
    """
    if output_path is None:
        output_path = Config.SUBMISSION_PATH

    submission = pd.DataFrame({"id": test_df["id"]})

    for target in Config.TARGET_COLS:
        if target in predictions_dict:
            submission[target] = predictions_dict[target]
        else:
            raise ValueError(f"Missing predictions for target: {target}")

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print("Head of submission:")
    print(submission.head())


def run_physics_informed_pipeline(train_df, val_df, test_df):
    """
    Orchestrates the training and prediction process for all targets.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.

    Returns:
        dict: Dictionary of trained models.
    """
    trained_models = {}
    all_predictions = {}

    print("\n--- Starting Physics-Informed XGBoost Training ---")

    for target in Config.TARGET_COLS:
        # Train
        model = train_target_model(train_df, val_df, target, verbose=True)
        trained_models[target] = model

        # Predict
        preds = predict_target(model, test_df)
        all_predictions[target] = preds

    # Generate Submission
    generate_submission_file(test_df, all_predictions)

    return trained_models
