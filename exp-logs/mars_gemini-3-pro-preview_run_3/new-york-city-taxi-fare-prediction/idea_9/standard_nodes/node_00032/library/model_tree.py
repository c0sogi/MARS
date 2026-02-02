import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from library.config import XGB_PARAMS, LGBM_PARAMS, WORKING_DIR, SEED
from library.utils import seed_everything

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)


def _prepare_xy(df, target_col="fare_amount", drop_cols=None):
    """
    Helper to separate features and target.
    """
    if drop_cols is None:
        drop_cols = ["key", "pickup_datetime"]

    # Identify features: all columns except target and excluded cols
    features = [c for c in df.columns if c != target_col and c not in drop_cols]

    X = df[features]
    y = df[target_col] if target_col in df.columns else None

    return X, y


def train_xgboost(df_train, df_val, load_cached_model=True):
    """
    Trains an XGBoost model or loads it from cache.

    Args:
        df_train (pd.DataFrame): Training data.
        df_val (pd.DataFrame): Validation data.
        load_cached_model (bool): Whether to try loading a saved model.

    Returns:
        xgb.XGBRegressor: Trained model.
    """
    seed_everything(SEED)
    model_path = os.path.join(WORKING_DIR, "xgboost_model.joblib")

    if load_cached_model and os.path.exists(model_path):
        print(f"Loading XGBoost model from {model_path}...")
        return joblib.load(model_path)

    print("Training XGBoost model...")
    X_train, y_train = _prepare_xy(df_train)
    X_val, y_val = _prepare_xy(df_val)

    # Prepare parameters
    params = XGB_PARAMS.copy()
    # early_stopping_rounds is passed to constructor in newer XGBoost versions

    # Initialize model
    model = xgb.XGBRegressor(**params)

    # Fit model
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate
    preds_val = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds_val))
    print(f"XGBoost Validation RMSE: {rmse}")

    # Save model
    print(f"Saving XGBoost model to {model_path}...")
    joblib.dump(model, model_path)

    return model


def train_lgbm(df_train, df_val, load_cached_model=True):
    """
    Trains a LightGBM model or loads it from cache.

    Args:
        df_train (pd.DataFrame): Training data.
        df_val (pd.DataFrame): Validation data.
        load_cached_model (bool): Whether to try loading a saved model.

    Returns:
        lgb.LGBMRegressor: Trained model.
    """
    seed_everything(SEED)
    model_path = os.path.join(WORKING_DIR, "lgbm_model.joblib")

    if load_cached_model and os.path.exists(model_path):
        print(f"Loading LightGBM model from {model_path}...")
        return joblib.load(model_path)

    print("Training LightGBM model...")
    X_train, y_train = _prepare_xy(df_train)
    X_val, y_val = _prepare_xy(df_val)

    # Prepare parameters
    params = LGBM_PARAMS.copy()
    early_stopping_rounds = params.pop("early_stopping_rounds", 100)

    # Initialize model
    model = lgb.LGBMRegressor(**params)

    # Setup callbacks for early stopping
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0),  # Suppress logging
    ]

    # Fit model
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=callbacks,
    )

    # Evaluate
    preds_val = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds_val))
    print(f"LightGBM Validation RMSE: {rmse}")

    # Save model
    print(f"Saving LightGBM model to {model_path}...")
    joblib.dump(model, model_path)

    return model


def predict_tree_model(model, df):
    """
    Generates predictions using a trained tree model.

    Args:
        model: Trained XGBoost or LightGBM model.
        df (pd.DataFrame): Data to predict on.

    Returns:
        np.array: Predictions.
    """
    X, _ = _prepare_xy(df)
    return model.predict(X)
