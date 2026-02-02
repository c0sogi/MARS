import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from library.config import (
    XGB_PARAMS,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    TARGET_COLS,
    SUBMISSION_PATH,
    WORKING_DIR,
)
from library.data_handler import (
    build_feature_matrix,
    load_metadata,
    log_transform_targets,
    inverse_transform_targets,
)


def train_xgboost(X_train, y_train, X_val, y_val, target_name):
    """
    Trains an XGBoost regressor for a specific target variable.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets (log-transformed).
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation targets (log-transformed).
        target_name (str): Name of the target variable for logging.

    Returns:
        xgb.XGBRegressor: The trained model.
    """
    print(f"\n--- Training XGBoost for {target_name} ---")

    # Initialize model with parameters from config
    model = xgb.XGBRegressor(**XGB_PARAMS)

    # Fit model with early stopping
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=VERBOSE_EVAL,
    )

    # Evaluate on validation set
    val_preds = model.predict(X_val)
    mse = mean_squared_error(y_val, val_preds)
    rmse = np.sqrt(mse)

    # Print full precision metric
    print(f"Validation RMSE (Log Scale) for {target_name}: {rmse}")

    return model


def make_predictions(model, X_test):
    """
    Generates predictions using a trained model.

    Args:
        model (xgb.XGBRegressor): Trained XGBoost model.
        X_test (pd.DataFrame): Test features.

    Returns:
        np.array: Predicted values (log-scale).
    """
    return model.predict(X_test)


def run_training_pipeline(sample_size=None, load_cached_data=True):
    """
    Orchestrates the data loading, preprocessing, and training process.

    Args:
        sample_size (int, optional): Limit dataset size for debugging.
        load_cached_data (bool): Whether to load features from cache.

    Returns:
        tuple: (dict of trained models, list of feature columns)
    """
    print(
        f"Starting training pipeline (Sample Size: {sample_size}, Cache: {load_cached_data})"
    )

    # 1. Load Metadata
    train_meta = load_metadata("train", sample_size=sample_size)
    val_meta = load_metadata("val", sample_size=sample_size)

    # 2. Build Feature Matrices (Features are extracted and cached here)
    # The CASH strategy features (RDF, ADF, Physical) are computed in build_feature_matrix
    df_train = build_feature_matrix(
        train_meta, "train", load_cached_data=load_cached_data
    )
    df_val = build_feature_matrix(val_meta, "val", load_cached_data=load_cached_data)

    # 3. Log Transform Targets
    # We predict log(1+y) to handle the wide range of energy values
    df_train_trans = log_transform_targets(df_train, TARGET_COLS)
    df_val_trans = log_transform_targets(df_val, TARGET_COLS)

    # 4. Prepare Feature Columns
    # Exclude ID and Targets from features
    exclude_cols = TARGET_COLS + ["id"]
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    print(f"Training with {len(feature_cols)} features.")

    X_train = df_train_trans[feature_cols]
    X_val = df_val_trans[feature_cols]

    models = {}

    # 5. Train a model for each target
    for target in TARGET_COLS:
        y_train = df_train_trans[target]
        y_val = df_val_trans[target]

        model = train_xgboost(X_train, y_train, X_val, y_val, target)
        models[target] = model

    return models, feature_cols


def generate_submission_file(
    models, feature_cols, sample_size=None, load_cached_data=True
):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        models (dict): Dictionary of trained models keyed by target name.
        feature_cols (list): List of feature names used during training.
        sample_size (int, optional): Limit test set size for debugging.
        load_cached_data (bool): Whether to load test features from cache.
    """
    print("\n--- Generating Submission ---")

    # 1. Load Test Data
    test_meta = load_metadata("test", sample_size=sample_size)
    df_test = build_feature_matrix(test_meta, "test", load_cached_data=load_cached_data)

    # 2. Prepare Test Features
    # Ensure columns match training data exactly
    # Fill missing columns with 0 if any (though build_feature_matrix should be consistent)
    X_test = df_test[feature_cols]

    # 3. Generate Predictions
    submission_df = pd.DataFrame({"id": df_test["id"]})

    for target in TARGET_COLS:
        if target in models:
            print(f"Predicting {target}...")
            # Predict in log scale
            log_preds = make_predictions(models[target], X_test)
            # Inverse transform to original scale
            orig_preds = inverse_transform_targets(log_preds)
            submission_df[target] = orig_preds
        else:
            print(f"Warning: No model found for {target}, filling with zeros.")
            submission_df[target] = 0.0

    # 4. Save Submission
    # Ensure directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print("First 5 rows of submission:")
    print(submission_df.head())
