import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from library.config import XGB_PARAMS, TARGET_COLS, SUBMISSION_PATH, RANDOM_SEED
from library.utils import log_transform, inverse_log_transform
from library.feature_processor import process_dataset
from library.data_loader import load_metadata


def train_xgboost(load_cached_data=True, debug=False):
    """
    Trains XGBoost models for each target variable (formation energy and bandgap energy).

    This function orchestrates the loading of features, merging with ground truth labels,
    applying log-transformations, and training separate regressors for each target.

    Args:
        load_cached_data (bool): If True, attempts to load pre-computed features from Parquet files.
        debug (bool): If True, uses a smaller subset of the data for rapid testing.

    Returns:
        dict: A dictionary where keys are target names and values are the trained XGBoost models.
    """
    # 1. Load Feature Data
    print(f"Loading training features (debug={debug})...")
    train_features_df = process_dataset(
        "train", load_cached_data=load_cached_data, debug=debug
    )

    print(f"Loading validation features (debug={debug})...")
    val_features_df = process_dataset(
        "val", load_cached_data=load_cached_data, debug=debug
    )

    # 2. Load Metadata (to get Targets)
    # We reload metadata to ensure we have the target labels associated with the IDs
    print("Loading metadata targets...")
    train_meta_df = load_metadata("train", debug=debug)
    val_meta_df = load_metadata("val", debug=debug)

    # 3. Merge Features with Targets
    # Ensure alignment by merging on 'id'
    train_merged = train_features_df.merge(
        train_meta_df[["id"] + TARGET_COLS], on="id", how="inner"
    )
    val_merged = val_features_df.merge(
        val_meta_df[["id"] + TARGET_COLS], on="id", how="inner"
    )

    # Define input feature columns (exclude targets, id, and file paths)
    exclude_cols = TARGET_COLS + ["id", "file_path"]
    feature_cols = [c for c in train_merged.columns if c not in exclude_cols]

    print(f"Training with {len(feature_cols)} features.")

    X_train = train_merged[feature_cols]
    X_val = val_merged[feature_cols]

    models = {}

    # 4. Train One Model Per Target
    for target in TARGET_COLS:
        print(f"\n{'='*40}")
        print(f"Training XGBoost for Target: {target}")
        print(f"{'='*40}")

        # Apply Log Transformation to Targets
        y_train = log_transform(train_merged[target].values)
        y_val = log_transform(val_merged[target].values)

        # Prepare Model Parameters
        # Pass early_stopping_rounds to constructor for XGBoost >= 1.6
        params = XGB_PARAMS.copy()
        if "early_stopping_rounds" not in params:
            params["early_stopping_rounds"] = 50

        model = xgb.XGBRegressor(**params)

        # Fit Model
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=100,
        )

        # 5. Evaluation
        # Predict on validation set (Log Scale)
        y_pred_log = model.predict(X_val)
        rmse_log = np.sqrt(mean_squared_error(y_val, y_pred_log))

        # Predict on validation set (Original Scale)
        y_pred_orig = inverse_log_transform(y_pred_log)
        y_val_orig = val_merged[target].values
        rmse_orig = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))

        print(f"\n>>> Validation RMSE (Log Scale / RMSLE): {rmse_log:.9f}")
        print(f">>> Validation RMSE (Original Scale):    {rmse_orig:.9f}")

        models[target] = model

    return models


def predict(models, load_cached_data=True, debug=False):
    """
    Generates predictions for the test set using the trained models.

    Args:
        models (dict): Dictionary of trained models keyed by target name.
        load_cached_data (bool): Whether to use cached test features.
        debug (bool): Debug mode flag.
    """
    print("\n" + "=" * 40)
    print("Generating Predictions for Test Set")
    print("=" * 40)

    # 1. Load Test Features
    test_features_df = process_dataset(
        "test", load_cached_data=load_cached_data, debug=debug
    )

    # 2. Align Features
    # Use the feature names from the first model to ensure correct column order
    first_target = TARGET_COLS[0]
    feature_names = models[first_target].get_booster().feature_names

    # Reindex test dataframe to match training features (handles missing/extra cols if any)
    X_test = test_features_df.reindex(columns=feature_names)

    # 3. Generate Predictions
    submission_df = pd.DataFrame()
    submission_df["id"] = test_features_df["id"]

    for target in TARGET_COLS:
        print(f"Predicting {target}...")
        model = models[target]

        # Predict in log space
        y_pred_log = model.predict(X_test)

        # Inverse transform to original space
        y_pred = inverse_log_transform(y_pred_log)

        submission_df[target] = y_pred

    # 4. Save Submission
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print(f"\nSubmission file saved successfully to: {SUBMISSION_PATH}")
    print("Head of submission:")
    print(submission_df.head())
