import os
import joblib
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from library.config import Config
from library.utils import setup_logger, compute_mcc

# Initialize logger
logger = setup_logger("model_zoo")


class LGBMWrapper:
    """
    Wrapper for LightGBM models to standardize training, prediction, and saving.
    Supports Scout (lightweight) and Expert (high-capacity) configurations.
    """

    def __init__(self, params, model_name="lgbm_model"):
        self.params = params.copy()
        self.model_name = model_name
        self.model = None
        # Extract n_estimators to use as num_boost_round
        self.n_estimators = self.params.pop("n_estimators", 1000)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the LightGBM model using lgb.train API.
        """
        logger.info(f"Training {self.model_name} with {self.n_estimators} rounds...")

        # Create LightGBM Datasets
        dtrain = lgb.Dataset(X_train, label=y_train)
        valid_sets = [dtrain]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            valid_sets.append(dval)
            valid_names.append("valid")

        # Configure callbacks
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=False
            ),
            lgb.log_evaluation(period=100),
        ]

        # Train model
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # Log final validation metric
        if X_val is not None and y_val is not None:
            preds = self.model.predict(X_val, num_iteration=self.model.best_iteration)
            # Compute MCC at default 0.5 threshold for logging purposes
            preds_bin = (preds > 0.5).astype(int)
            mcc = compute_mcc(y_val, preds_bin)
            logger.info(f"[{self.model_name}] Final Validation MCC: {mcc}")

    def predict(self, X):
        """
        Generates probability predictions.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def save(self):
        """
        Saves the model to disk using joblib.
        """
        path = os.path.join(Config.WORKING_DIR, f"{self.model_name}.joblib")
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load(self):
        """
        Loads the model from disk.
        """
        path = os.path.join(Config.WORKING_DIR, f"{self.model_name}.joblib")
        if os.path.exists(path):
            self.model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
        else:
            raise FileNotFoundError(f"Model file not found at {path}")


class XGBWrapper:
    """
    Wrapper for XGBoost models to standardize training, prediction, and saving.
    Designed for the Expert ensemble stage.
    """

    def __init__(self, params, model_name="xgb_model"):
        self.params = params.copy()
        self.model_name = model_name
        self.model = None
        # Extract n_estimators to use as num_boost_round
        self.n_estimators = self.params.pop("n_estimators", 1000)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the XGBoost model using xgb.train API.
        """
        logger.info(f"Training {self.model_name} with {self.n_estimators} rounds...")

        # Create DMatrix
        # XGBoost handles DataFrames correctly in DMatrix, preserving feature names
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, "train")]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "valid"))

        # Train model
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose_eval=100,
        )

        # Log final validation metric
        if X_val is not None and y_val is not None:
            # Predict on validation set
            preds = self.model.predict(dval)
            # Compute MCC at default 0.5 threshold for logging purposes
            preds_bin = (preds > 0.5).astype(int)
            mcc = compute_mcc(y_val, preds_bin)
            logger.info(f"[{self.model_name}] Final Validation MCC: {mcc}")

    def predict(self, X):
        """
        Generates probability predictions.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        # Convert to DMatrix for prediction
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def save(self):
        """
        Saves the model to disk using joblib.
        """
        path = os.path.join(Config.WORKING_DIR, f"{self.model_name}.joblib")
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load(self):
        """
        Loads the model from disk.
        """
        path = os.path.join(Config.WORKING_DIR, f"{self.model_name}.joblib")
        if os.path.exists(path):
            self.model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
        else:
            raise FileNotFoundError(f"Model file not found at {path}")
