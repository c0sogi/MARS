import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from library.config import XGB_PARAMS, TARGET_COLS, SUBMISSION_PATH, RANDOM_SEED
from library.data_handler import transform_targets, inverse_transform_targets
from library.feature_extractor import build_dataset

# Set global random seed for reproducibility
np.random.seed(RANDOM_SEED)


def train_model(
    load_cached_data: bool = True, n_estimators: int = None, limit_data: int = None
):
    """
    Trains XGBoost models for each target variable using the processed dataset.

    Args:
        load_cached_data (bool): Whether to load features from parquet cache.
        n_estimators (int, optional): Override the number of estimators in config.
        limit_data (int, optional): Limit the number of training samples for debugging.

    Returns:
        tuple: (models_dict, metrics_dict)
            models_dict: Dictionary mapping target names to trained XGBRegressor objects.
            metrics_dict: Dictionary mapping target names to validation RMSLE scores.
    """
    print("Loading training data...")
    train_df = build_dataset(split="train", load_cached_data=load_cached_data)

    print("Loading validation data...")
    val_df = build_dataset(split="val", load_cached_data=load_cached_data)

    # Apply data limit if requested (for debugging/fast runs)
    if limit_data is not None:
        print(f"Limiting training data to {limit_data} samples.")
        train_df = train_df.iloc[:limit_data]

    # Identify feature columns: All columns except ID and Targets
    # Note: 'file_path' is already dropped by build_dataset
    feature_cols = [c for c in train_df.columns if c not in TARGET_COLS + ["id"]]

    # Prepare X (Features)
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]

    # Prepare y (Targets) - Log Transformed
    # We transform the entire dataframe subset to ensure alignment
    y_train_log = transform_targets(train_df)
    y_val_log = transform_targets(val_df)

    models = {}
    metrics = {}

    # Update hyperparameters if overrides provided
    params = XGB_PARAMS.copy()
    if n_estimators is not None:
        params["n_estimators"] = n_estimators

    print(f"Training with features: {len(feature_cols)} dimensions.")

    for target in TARGET_COLS:
        print(f"\n--- Training model for Target: {target} ---")

        model = xgb.XGBRegressor(**params)

        # Train with early stopping
        model.fit(
            X_train,
            y_train_log[target],
            eval_set=[(X_val, y_val_log[target])],
            early_stopping_rounds=50,
            verbose=False,
        )

        # Validation evaluation
        preds_log = model.predict(X_val)

        # Calculate RMSE on the log-transformed scale (which is RMSLE on original scale)
        rmse_log = np.sqrt(mean_squared_error(y_val_log[target], preds_log))

        print(f"Validation RMSLE for {target}: {rmse_log}")

        models[target] = model
        metrics[target] = rmse_log

    return models, metrics


def generate_predictions(models: dict, load_cached_data: bool = True):
    """
    Generates predictions for the test set using trained models and saves to CSV.

    Args:
        models (dict): Dictionary of trained models keyed by target name.
        load_cached_data (bool): Whether to load test features from cache.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    print("\n--- Generating Predictions ---")
    print("Loading test data...")
    test_df = build_dataset(split="test", load_cached_data=load_cached_data)

    # Identify feature columns (ensure consistency with training)
    # Test data does not contain targets, so we just exclude 'id'
    feature_cols = [c for c in test_df.columns if c != "id"]
    X_test = test_df[feature_cols]

    submission_data = {"id": test_df["id"]}

    for target in TARGET_COLS:
        if target not in models:
            raise ValueError(f"No trained model found for target: {target}")

        print(f"Predicting {target}...")
        model = models[target]

        # Predict in log space
        preds_log = model.predict(X_test)

        # Inverse transform to original space
        preds_original = inverse_transform_targets(preds_log)

        submission_data[target] = preds_original

    # Construct DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Save to disk
    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    return submission_df
