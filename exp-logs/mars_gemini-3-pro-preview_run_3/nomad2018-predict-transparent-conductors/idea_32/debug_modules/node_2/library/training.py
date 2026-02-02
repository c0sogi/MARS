import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.config import XGB_PARAMS, TARGET_COLS, RANDOM_SEED


class DualTargetRegressor:
    """
    Wrapper class to handle training and prediction for two separate targets
    (Formation Energy and Bandgap Energy) using XGBoost with log-transformation.
    """

    def __init__(self, params=None, n_estimators=None):
        """
        Args:
            params (dict): XGBoost hyperparameters. If None, uses defaults from config.
            n_estimators (int): Optional override for number of estimators (e.g. for quick debugging).
        """
        self.params = params.copy() if params else XGB_PARAMS.copy()

        if n_estimators is not None:
            self.params["n_estimators"] = n_estimators

        self.models = {}
        for target in TARGET_COLS:
            self.models[target] = xgb.XGBRegressor(**self.params)

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        early_stopping_rounds=100,
        verbose=False,
    ):
        """
        Fits the models for both targets. Applies log1p transformation to targets internally.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.DataFrame): Training targets.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.DataFrame): Validation targets.
            early_stopping_rounds (int): Rounds for early stopping.
            verbose (bool): Verbosity of training.
        """
        # Transform targets: z = log(1 + y)
        y_train_log = np.log1p(y_train[TARGET_COLS])

        y_val_log = None
        if y_val is not None:
            y_val_log = np.log1p(y_val[TARGET_COLS])

        for target in TARGET_COLS:
            if verbose:
                print(f"Training model for target: {target}")

            model = self.models[target]

            eval_set = None
            if X_val is not None and y_val_log is not None:
                eval_set = [(X_train, y_train_log[target]), (X_val, y_val_log[target])]

            # Cite debug_lesson_1: Update XGBoost Early Stopping Syntax for Versions 1.6+
            # Pass early_stopping_rounds via set_params instead of fit
            if eval_set:
                model.set_params(early_stopping_rounds=early_stopping_rounds)
            else:
                model.set_params(early_stopping_rounds=None)

            model.fit(
                X_train,
                y_train_log[target],
                eval_set=eval_set,
                verbose=verbose,
            )

    def predict(self, X):
        """
        Predicts targets for input X. Applies expm1 inverse transformation.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame: Predicted values in original scale.
        """
        predictions = {}
        for target in TARGET_COLS:
            model = self.models[target]
            # Predict log-transformed values
            pred_log = model.predict(X)
            # Inverse transform: y = exp(z) - 1
            # Clip to avoid overflow/underflow issues if any, though unlikely with these physical values
            pred_original = np.expm1(pred_log)
            # Ensure non-negative predictions as energies are non-negative
            predictions[target] = np.maximum(pred_original, 0)

        return pd.DataFrame(predictions, index=X.index)


def train_and_evaluate(X_train, y_train, X_val, y_val, params=None, n_estimators=None):
    """
    Trains the DualTargetRegressor and evaluates on validation set.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        params: Hyperparameters for XGBoost.
        n_estimators: Override for number of trees.

    Returns:
        model: Trained DualTargetRegressor instance.
        metrics: Dictionary of metrics.
    """
    print(f"Initializing DualTargetRegressor with n_estimators={n_estimators}...")
    model = DualTargetRegressor(params=params, n_estimators=n_estimators)

    print("Fitting models with early stopping...")
    model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=50, verbose=False)

    print("Generating validation predictions...")
    preds_val = model.predict(X_val)

    # Calculate Metrics
    # Metric 1: RMSLE (Root Mean Squared Logarithmic Error)
    # Since we trained on log1p, RMSE of log-transformed values is the RMSLE.
    # However, let's calculate it explicitly from the original scale predictions to be sure.
    # RMSLE = sqrt(mean((log1p(pred) - log1p(true))^2))

    metrics = {}

    print("\n--- Validation Metrics ---")
    for target in TARGET_COLS:
        y_true = y_val[target]
        y_pred = preds_val[target]

        # RMSLE
        rmsle = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

        # RMSE (Original Scale)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        metrics[f"rmsle_{target}"] = rmsle
        metrics[f"rmse_{target}"] = rmse

        print(f"Target: {target}")
        print(f"  RMSLE: {rmsle}")
        print(f"  RMSE:  {rmse}")

    # Aggregate RMSLE (Column-wise mean)
    avg_rmsle = np.mean([metrics[f"rmsle_{t}"] for t in TARGET_COLS])
    print(f"\nAverage RMSLE: {avg_rmsle}")

    return model, metrics
