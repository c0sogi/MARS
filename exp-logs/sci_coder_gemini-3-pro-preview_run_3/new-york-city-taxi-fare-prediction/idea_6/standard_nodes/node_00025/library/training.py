import os
import gc
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

from library.config import Config
from library.data_processing import process_data
from library.model_factory import ModelFactory


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Trains an XGBoost model using the configuration provided in Config.
    """
    print("Initializing XGBoost model...")
    model = ModelFactory.create_xgboost(
        early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS
    )

    print(f"Training XGBoost on {len(X_train)} samples with GPU...")

    # XGBoost fit with early stopping
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=Config.VERBOSE_EVAL,
    )

    return model


def train_lgbm(X_train, y_train, X_val, y_val):
    """
    Trains a LightGBM model using the configuration provided in Config.
    """
    print("Initializing LightGBM model...")
    model = ModelFactory.create_lgbm()

    print(f"Training LightGBM on {len(X_train)} samples with CPU...")

    # LightGBM fit with callbacks for early stopping and logging
    callbacks = [
        lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(period=Config.VERBOSE_EVAL),
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=callbacks,
    )

    return model


def evaluate_model(model, X_val, y_val, model_name):
    """
    Predicts on validation set and calculates RMSE.
    """
    print(f"Evaluating {model_name}...")
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"{model_name} Validation RMSE: {rmse}")
    return preds, rmse


def generate_submission(models, test_df, feature_cols):
    """
    Generates predictions for the test set using the ensemble weights
    and saves the submission file.
    """
    print("Generating predictions for test set...")

    X_test = test_df[feature_cols]
    final_preds = np.zeros(len(X_test))

    # Calculate weighted average
    for model_name, model in models.items():
        weight = Config.ENSEMBLE_WEIGHTS.get(model_name, 0.0)
        if weight > 0:
            print(f"Predicting with {model_name} (Weight: {weight})...")
            preds = model.predict(X_test)
            final_preds += preds * weight

    # Create submission DataFrame
    submission = pd.DataFrame({"key": test_df["key"], "fare_amount": final_preds})

    # Save submission
    save_path = Config.SUBMISSION_FILE_PATH
    print(f"Saving submission to {save_path}...")
    submission.to_csv(save_path, index=False)
    print("Submission saved successfully.")


def run_training_pipeline(
    load_cached_data=True, debug_sample_size=Config.DEBUG_SAMPLE_SIZE
):
    """
    Main orchestration function for the training pipeline.

    Args:
        load_cached_data (bool): Whether to load processed data from cache.
        debug_sample_size (int, optional): Size of data to use for debugging.
    """
    # 1. Load and Process Data
    train_df, val_df, test_df = process_data(
        load_cached_data=load_cached_data, debug_sample_size=debug_sample_size
    )

    # 2. Prepare Feature Columns
    # Exclude target and ID columns
    exclude_cols = [
        "key",
        "fare_amount",
        "pickup_datetime",
    ]  # pickup_datetime dropped in processing but just in case
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    print(f"Training with features: {feature_cols}")

    X_train = train_df[feature_cols]
    y_train = train_df["fare_amount"]
    X_val = val_df[feature_cols]
    y_val = val_df["fare_amount"]

    # Clean up dataframe memory if possible, though we need columns for X
    # We keep df references to avoid immediate GC of underlying arrays if views are used

    trained_models = {}

    # 3. Train XGBoost
    if Config.ENSEMBLE_WEIGHTS.get("xgb", 0) > 0:
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val)

        # Save model
        xgb_path = os.path.join(Config.WORKING_DIR, "xgboost_model.joblib")
        joblib.dump(xgb_model, xgb_path)
        print(f"XGBoost model saved to {xgb_path}")

        evaluate_model(xgb_model, X_val, y_val, "XGBoost")
        trained_models["xgb"] = xgb_model

        # Force garbage collection
        gc.collect()

    # 4. Train LightGBM
    if Config.ENSEMBLE_WEIGHTS.get("lgbm", 0) > 0:
        lgbm_model = train_lgbm(X_train, y_train, X_val, y_val)

        # Save model
        lgbm_path = os.path.join(Config.WORKING_DIR, "lgbm_model.joblib")
        joblib.dump(lgbm_model, lgbm_path)
        print(f"LightGBM model saved to {lgbm_path}")

        evaluate_model(lgbm_model, X_val, y_val, "LightGBM")
        trained_models["lgbm"] = lgbm_model

        gc.collect()

    # 5. Generate Submission
    if trained_models:
        generate_submission(trained_models, test_df, feature_cols)
    else:
        print("No models were trained. Check ENSEMBLE_WEIGHTS in Config.")
