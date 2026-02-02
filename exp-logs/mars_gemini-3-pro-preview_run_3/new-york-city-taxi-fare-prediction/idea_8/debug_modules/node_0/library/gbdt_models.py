import os
import joblib
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from library import config, utils


def train_xgboost(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Trains an XGBoost Regressor using the provided training and validation dataframes.

    Args:
        train_df (pd.DataFrame): Training data containing features and target.
        val_df (pd.DataFrame): Validation data containing features and target.

    Returns:
        tuple: (trained_model, validation_predictions)
    """
    utils.seed_everything(config.SEED)

    # Select features and target
    features = config.TREE_FEATURES
    target = "fare_amount"

    # Prepare X and y
    X_train = train_df[features]
    y_train = train_df[target]

    X_val = val_df[features]
    y_val = val_df[target]

    print(f"\n=== Training XGBoost ===")
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    print(f"Features: {len(features)}")
    print(f"Device: {config.XGB_PARAMS.get('tree_method', 'auto')}")

    # Initialize model with config parameters
    model = xgb.XGBRegressor(**config.XGB_PARAMS)

    # Fit model with early stopping
    # Note: early_stopping_rounds is passed to fit() for compatibility with standard sklearn wrapper usage
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=100,
    )

    # Save model
    os.makedirs(os.path.dirname(config.MODEL_XGB_PATH), exist_ok=True)
    model.save_model(config.MODEL_XGB_PATH)
    print(f"XGBoost model saved to {config.MODEL_XGB_PATH}")

    # Generate validation predictions
    val_preds = model.predict(X_val)

    # Calculate and print metrics
    rmse = utils.compute_rmse(y_val, val_preds)
    print(f"XGBoost Validation RMSE: {rmse}")

    return model, val_preds


def train_lgbm(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Trains a LightGBM Regressor using the provided training and validation dataframes.

    Args:
        train_df (pd.DataFrame): Training data containing features and target.
        val_df (pd.DataFrame): Validation data containing features and target.

    Returns:
        tuple: (trained_model, validation_predictions)
    """
    utils.seed_everything(config.SEED)

    # Select features and target
    features = config.TREE_FEATURES
    target = "fare_amount"

    # Prepare X and y
    X_train = train_df[features]
    y_train = train_df[target]

    X_val = val_df[features]
    y_val = val_df[target]

    print(f"\n=== Training LightGBM ===")
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    print(f"Features: {len(features)}")

    # Initialize model with config parameters
    model = lgb.LGBMRegressor(**config.LGBM_PARAMS)

    # Define callbacks for early stopping and logging
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    # Fit model
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=callbacks,
    )

    # Save model
    os.makedirs(os.path.dirname(config.MODEL_LGBM_PATH), exist_ok=True)
    # Save using the booster's native save method for text format
    model.booster_.save_model(config.MODEL_LGBM_PATH)
    print(f"LightGBM model saved to {config.MODEL_LGBM_PATH}")

    # Generate validation predictions
    val_preds = model.predict(X_val)

    # Calculate and print metrics
    rmse = utils.compute_rmse(y_val, val_preds)
    print(f"LightGBM Validation RMSE: {rmse}")

    return model, val_preds
