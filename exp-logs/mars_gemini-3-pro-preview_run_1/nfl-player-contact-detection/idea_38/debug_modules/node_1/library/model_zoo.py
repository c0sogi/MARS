import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import os
from library.config import Config
from library.utils import save_joblib, load_joblib


class LGBMWrapper:
    """
    Wrapper for LightGBM model with specific handling for the NFL Contact Detection task.
    Uses parameters defined in Config.LGBM_PARAMS.
    """

    def __init__(self):
        self.params = Config.LGBM_PARAMS.copy()
        self.model = None
        self.best_iteration = 0

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Trains the LightGBM model with early stopping.
        """
        # Create LightGBM datasets
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Train
        self.model = lgb.train(
            self.params,
            dtrain,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=False
                ),
                lgb.log_evaluation(period=0),  # Suppress verbose output
            ],
        )

        self.best_iteration = self.model.best_iteration

        # Manually print the result of the best iteration
        # Note: LightGBM metrics are stored in model.best_score
        train_loss = self.model.best_score["train"]["binary_logloss"]
        val_loss = self.model.best_score["valid"]["binary_logloss"]
        print(f"[LGBM] Best Iteration: {self.best_iteration}")
        print(f"[LGBM] Best Train LogLoss: {train_loss}")
        print(f"[LGBM] Best Valid LogLoss: {val_loss}")

    def predict_proba(self, X):
        """
        Predicts probabilities using the trained model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X, num_iteration=self.best_iteration)

    def save(self, filename):
        """
        Saves the model wrapper to disk.
        """
        save_joblib(self, filename)

    @staticmethod
    def load(filename):
        """
        Loads the model wrapper from disk.
        """
        return load_joblib(filename)


class XGBWrapper:
    """
    Wrapper for XGBoost model with dynamic scale_pos_weight calculation.
    Uses parameters defined in Config.XGB_PARAMS.
    """

    def __init__(self):
        self.params = Config.XGB_PARAMS.copy()
        self.model = None
        self.best_iteration = 0

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Trains the XGBoost model with early stopping.
        Dynamically calculates scale_pos_weight based on training data balance.
        """
        # Calculate scale_pos_weight
        # ratio = neg / pos
        n_pos = np.sum(y_train)
        n_neg = len(y_train) - n_pos
        ratio = n_neg / max(1, n_pos)
        self.params["scale_pos_weight"] = ratio

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Train
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params.get("n_estimators", 2000),
            evals=[(dtrain, "train"), (dval, "valid")],
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )

        self.best_iteration = self.model.best_iteration

        # Retrieve metrics
        # XGBoost stores evaluation history in model object but getting the exact best score requires lookup
        # We can predict on val to verify or just trust the internal state.
        # For logging purposes, let's just print the best iteration.
        print(f"[XGB] Best Iteration: {self.best_iteration}")
        print(f"[XGB] Calculated scale_pos_weight: {ratio}")
        # Note: XGBoost best_score attribute is the score of the best iteration
        print(f"[XGB] Best Score (LogLoss): {self.model.best_score}")

    def predict_proba(self, X):
        """
        Predicts probabilities using the trained model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest, iteration_range=(0, self.best_iteration + 1))

    def save(self, filename):
        """
        Saves the model wrapper to disk.
        """
        save_joblib(self, filename)

    @staticmethod
    def load(filename):
        """
        Loads the model wrapper from disk.
        """
        return load_joblib(filename)


class EnsemblePredictor:
    """
    Loads trained Expert LightGBM and XGBoost models and computes
    the unweighted average of their predictions.
    """

    def __init__(self):
        self.lgbm_model = None
        self.xgb_model = None

    def load_models(self, lgbm_path, xgb_path):
        """
        Loads the pre-trained models from the specified paths.
        """
        print(f"Loading LightGBM model from {lgbm_path}...")
        self.lgbm_model = LGBMWrapper.load(lgbm_path)
        if self.lgbm_model is None:
            raise FileNotFoundError(f"Failed to load LightGBM model from {lgbm_path}")

        print(f"Loading XGBoost model from {xgb_path}...")
        self.xgb_model = XGBWrapper.load(xgb_path)
        if self.xgb_model is None:
            raise FileNotFoundError(f"Failed to load XGBoost model from {xgb_path}")

    def predict(self, X):
        """
        Generates ensemble predictions (unweighted average).

        Args:
            X (pd.DataFrame): Feature matrix.

        Returns:
            np.array: Predicted probabilities.
        """
        if self.lgbm_model is None or self.xgb_model is None:
            raise ValueError("Models not loaded. Call load_models() first.")

        # Get predictions from individual models
        pred_lgbm = self.lgbm_model.predict_proba(X)
        pred_xgb = self.xgb_model.predict_proba(X)

        # Average
        pred_ensemble = (pred_lgbm + pred_xgb) / 2.0

        return pred_ensemble
