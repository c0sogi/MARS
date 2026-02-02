import os
import numpy as np
import pandas as pd
import xgboost as xgb
from library.config import WORKING_DIR, SUBMISSION_PATH, XGB_PARAMS, TRAIN_PARAMS
from library.feature_engineering import get_target_encoded_data


def train_model(train_df, val_df, params=None, train_config=None):
    """
    Trains an XGBoost model using the provided training and validation dataframes.

    Args:
        train_df (pd.DataFrame): Training data containing features and target.
        val_df (pd.DataFrame): Validation data containing features and target.
        params (dict, optional): XGBoost hyperparameters. Defaults to XGB_PARAMS from config.
        train_config (dict, optional): Training loop configuration. Defaults to TRAIN_PARAMS from config.

    Returns:
        tuple: (trained_model, list_of_feature_names)
    """
    # Set fixed seeds for reproducibility
    np.random.seed(42)

    # Use defaults if not provided
    if params is None:
        params = XGB_PARAMS.copy()
    if train_config is None:
        train_config = TRAIN_PARAMS.copy()

    target_col = "fare_amount"

    # Identify feature columns
    # Exclude ID, Target, and raw Datetime objects (which XGBoost cannot handle directly)
    exclude_cols = {"key", target_col, "pickup_datetime"}
    features = [c for c in train_df.columns if c not in exclude_cols]

    print(f"Starting training with {len(features)} features: {features}")
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # Create DMatrix objects
    # This is optimized for memory and speed
    dtrain = xgb.DMatrix(train_df[features], label=train_df[target_col])
    dval = xgb.DMatrix(val_df[features], label=val_df[target_col])

    # Extract num_boost_round from params if present, otherwise default
    num_boost_round = params.pop("n_estimators", 10000)

    # Define evaluation list for early stopping
    evals = [(dtrain, "train"), (dval, "eval")]

    # Train the model
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=train_config.get("early_stopping_rounds", 100),
        verbose_eval=train_config.get("verbose_eval", 100),
    )

    # Save the model
    model_path = os.path.join(WORKING_DIR, "xgb_model.json")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    return model, features


def evaluate_model(model, val_df, features):
    """
    Evaluates the model on the validation set and prints the RMSE.

    Args:
        model (xgb.Booster): Trained XGBoost model.
        val_df (pd.DataFrame): Validation data.
        features (list): List of feature names used for training.

    Returns:
        float: Root Mean Squared Error.
    """
    target_col = "fare_amount"

    # Create DMatrix for validation
    dval = xgb.DMatrix(val_df[features])

    # Generate predictions
    preds = model.predict(dval)
    actuals = val_df[target_col].values

    # Calculate RMSE
    mse = np.mean((preds - actuals) ** 2)
    rmse = np.sqrt(mse)

    # Print full precision as requested
    print(f"Validation RMSE: {rmse:.16f}")

    return rmse


def predict_and_submit(model, test_df, features, output_path=SUBMISSION_PATH):
    """
    Generates predictions for the test set, applies post-processing,
    and saves the submission file.

    Args:
        model (xgb.Booster): Trained XGBoost model.
        test_df (pd.DataFrame): Test data.
        features (list): List of feature names used for training.
        output_path (str): Path to save the CSV submission.
    """
    print("Generating predictions for test set...")

    # Create DMatrix for test
    dtest = xgb.DMatrix(test_df[features])

    # Generate predictions
    preds = model.predict(dtest)

    # Apply minimum fare floor (e.g., $2.50) as per Idea strategy
    # This handles potential under-predictions for very short trips
    preds = np.maximum(preds, 2.50)

    # Create submission DataFrame
    submission = pd.DataFrame({"key": test_df["key"], "fare_amount": preds})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training_pipeline(load_cached_data=True):
    """
    Orchestrates the full training pipeline:
    1. Loads and engineers data (Train/Val/Test).
    2. Trains the XGBoost model.
    3. Evaluates performance.
    4. Generates submission file.

    Args:
        load_cached_data (bool): Whether to attempt loading pre-processed data from cache.
    """
    print("=== Starting Training Pipeline ===")

    # 1. Load Data
    # The get_target_encoded_data function handles caching and feature engineering internally
    train_df = get_target_encoded_data("train", load_cached_data=load_cached_data)
    val_df = get_target_encoded_data("val", load_cached_data=load_cached_data)

    # 2. Train Model
    model, features = train_model(train_df, val_df)

    # 3. Evaluate
    evaluate_model(model, val_df, features)

    # 4. Generate Submission
    # Load test data
    test_df = get_target_encoded_data("test", load_cached_data=load_cached_data)
    predict_and_submit(model, test_df, features)

    print("=== Pipeline Complete ===")
