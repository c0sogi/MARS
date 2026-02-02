import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error

from library.config import (
    LGBM_PARAMS,
    SEED,
    N_FOLDS,
    SUBMISSION_PATH,
    MODEL_PATH,
    WORKING_DIR,
)
from library.feature_engineering import process_dataset


def train_lgbm(X_train, y_train, X_val, y_val, params):
    """
    Trains a LightGBM Regressor with early stopping.

    Args:
        X_train, y_train: Training data and targets.
        X_val, y_val: Validation data and targets for early stopping.
        params: Dictionary of LightGBM hyperparameters.

    Returns:
        Trained LGBMRegressor model.
    """
    # Create a copy of params to avoid modifying the global config
    train_params = params.copy()

    # Extract early_stopping_rounds as it is passed via callbacks in newer sklearn API
    early_stopping_rounds = train_params.pop("early_stopping_rounds", 100)

    # Initialize model
    model = lgb.LGBMRegressor(**train_params)

    # Configure callbacks: Early Stopping and Silent Logging
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    # Fit model
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=callbacks,
    )

    return model


def run_cross_validation(load_cached_data=True, debug=False):
    """
    Performs Stratified K-Fold Cross Validation on the combined train and val sets.

    Args:
        load_cached_data (bool): Whether to load features from parquet cache.
        debug (bool): If True, runs on a small subset of data.

    Returns:
        tuple: (list of trained models, average MAE score)
    """
    print("Loading training and validation data for Cross-Validation...")

    # Load features using the library function
    # We combine train and val splits to perform a full K-Fold CV
    df_train = process_dataset("train", load_cached_data=load_cached_data, debug=debug)
    df_val = process_dataset("val", load_cached_data=load_cached_data, debug=debug)

    full_df = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    # Separate Features and Target
    drop_cols = ["segment_id", "time_to_eruption"]
    feature_cols = [c for c in full_df.columns if c not in drop_cols]

    X = full_df[feature_cols]
    y = full_df["time_to_eruption"]

    # Create bins for StratifiedKFold since target is continuous
    # We use quantiles to ensure balanced bins
    num_bins = 10
    y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    models = []
    mae_scores = []

    print(f"Starting {N_FOLDS}-Fold Stratified Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
        # Split data
        X_t, y_t = X.iloc[train_idx], y.iloc[train_idx]
        X_v, y_v = X.iloc[val_idx], y.iloc[val_idx]

        # Train model
        model = train_lgbm(X_t, y_t, X_v, y_v, LGBM_PARAMS)

        # Evaluate
        y_pred = model.predict(X_v)
        mae = mean_absolute_error(y_v, y_pred)

        mae_scores.append(mae)
        models.append(model)

        print(f"Fold {fold + 1} MAE: {mae}")

    # Aggregate Results
    avg_mae = np.mean(mae_scores)
    print(f"Average CV MAE: {avg_mae}")

    # Save the best model (lowest MAE) to disk as per requirements
    best_fold_idx = np.argmin(mae_scores)
    best_model = models[best_fold_idx]

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    # Access the underlying booster to save in LightGBM text format
    best_model.booster_.save_model(MODEL_PATH)
    print(f"Best model (from Fold {best_fold_idx + 1}) saved to {MODEL_PATH}")

    return models, avg_mae


def predict_and_submit(models, load_cached_data=True, debug=False):
    """
    Generates predictions for the test set using the trained model ensemble.
    Saves the submission file to the specified path.

    Args:
        models (list): List of trained LightGBM models.
        load_cached_data (bool): Whether to load features from cache.
        debug (bool): If True, processes a subset of test data.
    """
    print("Loading test data...")
    df_test = process_dataset("test", load_cached_data=load_cached_data, debug=debug)

    # Prepare Test Features
    segment_ids = df_test["segment_id"]
    # Test set only has segment_id and features, no target
    feature_cols = [c for c in df_test.columns if c != "segment_id"]
    X_test = df_test[feature_cols]

    print("Generating predictions using model ensemble...")

    # Average predictions across all fold models
    final_preds = np.zeros(len(X_test))
    for model in models:
        final_preds += model.predict(X_test)

    final_preds /= len(models)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"segment_id": segment_ids, "time_to_eruption": final_preds}
    )

    # Save to CSV
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print(f"Submission saved to {SUBMISSION_PATH}")
