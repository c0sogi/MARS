import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from library.config import Config


def train_xgboost(X_train, y_train, X_val, y_val, verbose=True):
    """
    Trains separate XGBoost regressors for each target variable.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.DataFrame): Training targets (log-transformed).
        X_val (pd.DataFrame): Validation features.
        y_val (pd.DataFrame): Validation targets (log-transformed).
        verbose (bool): Whether to print training progress.

    Returns:
        dict: A dictionary mapping target names to trained XGBRegressor objects.
    """
    models = {}

    # Ensure inputs are dataframes to handle column names easily
    if not isinstance(y_train, pd.DataFrame):
        y_train = pd.DataFrame(y_train)
    if not isinstance(y_val, pd.DataFrame):
        y_val = pd.DataFrame(y_val, columns=y_train.columns)

    targets = y_train.columns.tolist()

    for target in targets:
        if verbose:
            print(f"Training model for target: {target}")

        # Initialize model with parameters from Config
        model = xgb.XGBRegressor(**Config.XGB_PARAMS)

        # Extract specific target series
        y_train_col = y_train[target]
        y_val_col = y_val[target]

        # Train with early stopping
        # Note: early_stopping_rounds is passed to fit, not init in recent sklearn API wrappers
        model.fit(
            X_train,
            y_train_col,
            eval_set=[(X_train, y_train_col), (X_val, y_val_col)],
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose=Config.VERBOSE_EVAL,
        )

        models[target] = model

        if verbose:
            # Best iteration score (RMSE on log data)
            # XGBoost sklearn API stores best_score if early stopping is used
            # If not available directly, we can look at evals_result
            try:
                print(f"Best RMSE (log-scale) for {target}: {model.best_score}")
            except AttributeError:
                pass

    return models


def predict_xgboost(models, X):
    """
    Generates predictions using the trained models.

    Args:
        models (dict): Dictionary of trained XGBRegressor objects.
        X (pd.DataFrame): Features to predict on.

    Returns:
        pd.DataFrame: DataFrame containing predictions for each target.
    """
    predictions = {}

    for target, model in models.items():
        # Predict
        pred = model.predict(X)
        predictions[target] = pred

    return pd.DataFrame(predictions, index=X.index)


def evaluate_model(y_true, y_pred):
    """
    Evaluates the model performance using RMSLE (calculated as RMSE on log-transformed data).

    Args:
        y_true (pd.DataFrame): Ground truth targets (log-transformed).
        y_pred (pd.DataFrame): Predicted targets (log-transformed).

    Returns:
        dict: Dictionary containing RMSE for each target and the mean column-wise RMSE.
    """
    metrics = {}
    total_rmse = 0.0
    count = 0

    # Align indices just in case
    y_true = y_true.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)

    for col in y_true.columns:
        if col in y_pred.columns:
            mse = mean_squared_error(y_true[col], y_pred[col])
            rmse = np.sqrt(mse)
            metrics[f"rmse_{col}"] = rmse
            total_rmse += rmse
            count += 1
            print(f"RMSE (log-scale) for {col}: {rmse}")

    if count > 0:
        mean_rmse = total_rmse / count
        metrics["mean_column_wise_rmse"] = mean_rmse
        print(f"Mean Column-wise RMSE (RMSLE): {mean_rmse}")

    return metrics
