import xgboost as xgb
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything
from library.data_factory import DataFactory


def train_xgboost_model(load_cached_data=True):
    """
    Trains an XGBoost model using the configuration and data factory.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.

    Returns:
        tuple: (booster, predict_fn)
            - booster: The trained XGBoost Booster object.
            - predict_fn: A function that takes a pandas DataFrame or DMatrix and returns predictions.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    print("Initializing XGBoost Trainer...")

    # 2. Data Loading
    # DataFactory handles caching and engineering
    data_dict = DataFactory.get_xgb_data(load_cached_data=load_cached_data)

    X_train, y_train = data_dict["train"]
    X_val, y_val = data_dict["val"]

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}")

    # 3. Create DMatrices
    # XGBoost on GPU with 'hist' is very efficient.
    print("Creating DMatrices...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # 4. Configuration
    params = Config.XGB_PARAMS.copy()
    fit_params = Config.XGB_FIT_PARAMS.copy()

    # 5. Training
    print("Starting Training...")
    evals_result = {}

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=fit_params["num_boost_round"],
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=fit_params["early_stopping_rounds"],
        verbose_eval=fit_params["verbose_eval"],
        evals_result=evals_result,
    )

    # 6. Logging Best Score
    best_iteration = booster.best_iteration
    metric = params.get("eval_metric", "mlogloss")

    # Retrieve best score from history
    if "val" in evals_result and metric in evals_result["val"]:
        # Access the score at the best iteration (index)
        best_score = evals_result["val"][metric][best_iteration]
        print(f"Training finished. Best Iteration: {best_iteration}")
        print(f"Best Validation {metric}: {best_score}")
    else:
        print("Training finished. (Metric not found in evals_result)")

    # 7. Define Prediction Function
    def predict_fn(data):
        """
        Generates predictions using the trained booster.

        Args:
            data: pandas DataFrame or xgb.DMatrix

        Returns:
            np.ndarray: Prediction probabilities
        """
        if isinstance(data, xgb.DMatrix):
            dmat = data
        else:
            # Create DMatrix from DataFrame/Array
            dmat = xgb.DMatrix(data)

        return booster.predict(dmat)

    return booster, predict_fn
