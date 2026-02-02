import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error
from library.config import XGB_PARAMS, LGBM_PARAMS


def train_xgboost_model(X_train, y_train, X_val, y_val):
    """
    Trains an XGBoost Regressor using parameters defined in config.
    Utilizes GPU acceleration.
    """
    print("Initializing XGBoost model...")

    # Prepare parameters
    params = XGB_PARAMS.copy()

    model = xgb.XGBRegressor(**params)

    print(f"Training XGBoost on {len(X_train)} samples with GPU...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,  # Print progress every 100 rounds
    )

    # Evaluate
    print("Evaluating XGBoost model...")
    # XGBoost's predict automatically uses the best iteration if early stopping occurred
    val_preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, val_preds, squared=False)
    print(f"XGBoost Validation RMSE: {rmse}")

    return model


def train_lightgbm_model(X_train, y_train, X_val, y_val):
    """
    Trains a LightGBM Regressor using parameters defined in config.
    Utilizes CPU multi-threading.
    """
    print("Initializing LightGBM model...")

    # Prepare parameters
    params = LGBM_PARAMS.copy()
    early_stopping_rounds = params.pop("early_stopping_rounds", 50)

    # Configure callbacks for early stopping and logging
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.LGBMRegressor(**params)

    print(f"Training LightGBM on {len(X_train)} samples with CPU...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=callbacks,
    )

    # Evaluate
    print("Evaluating LightGBM model...")
    # LightGBM's predict automatically uses the best iteration if early stopping occurred
    val_preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, val_preds, squared=False)
    print(f"LightGBM Validation RMSE: {rmse}")

    return model
