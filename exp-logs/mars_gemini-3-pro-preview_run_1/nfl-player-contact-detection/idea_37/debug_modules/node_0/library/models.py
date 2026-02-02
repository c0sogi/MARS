import os
import joblib
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("models")


class BaseModel:
    """
    Abstract base class for model wrappers to ensure consistent interface.
    """

    def __init__(self):
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the model with early stopping.
        """
        raise NotImplementedError

    def predict(self, X):
        """
        Predicts probabilities for the positive class.
        """
        raise NotImplementedError

    def save(self, path):
        """
        Saves the model artifact to disk.
        """
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load(self, path):
        """
        Loads the model artifact from disk.
        """
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")


class LGBMWrapper(BaseModel):
    def __init__(self):
        super().__init__()
        self.params = Config.LGBM_PARAMS.copy()
        self.n_estimators = Config.N_ESTIMATORS
        self.early_stopping_rounds = Config.EARLY_STOPPING_ROUNDS

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains a LightGBM model using the configured parameters.
        """
        logger.info("Training LightGBM model...")

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # callbacks
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.early_stopping_rounds, verbose=False
            ),
            lgb.log_evaluation(period=100),
        ]

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Log best score with full precision
        best_score = self.model.best_score["valid"]["binary_logloss"]
        logger.info(f"Best LightGBM Validation LogLoss: {best_score:.16f}")

        # Calculate MCC on validation set for reference
        preds_val = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        # Use a default threshold of 0.5 for reporting, though this will be tuned later
        preds_binary = (preds_val > 0.5).astype(int)
        mcc = matthews_corrcoef(y_val, preds_binary)
        logger.info(f"LightGBM Validation MCC (thresh=0.5): {mcc:.16f}")

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        return self.model.predict(X, num_iteration=self.model.best_iteration)


class XGBWrapper(BaseModel):
    def __init__(self):
        super().__init__()
        self.params = Config.XGB_PARAMS.copy()
        self.n_estimators = Config.N_ESTIMATORS
        self.early_stopping_rounds = Config.EARLY_STOPPING_ROUNDS

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains an XGBoost model using the configured parameters.
        """
        logger.info("Training XGBoost model...")

        # Create DMatrices
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Train
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=[(dtrain, "train"), (dval, "valid")],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=100,
        )

        # Log best score with full precision
        # XGBoost stores evaluation results in the object, but best_score attribute is available
        best_score = self.model.best_score
        logger.info(f"Best XGBoost Validation LogLoss: {best_score:.16f}")

        # Calculate MCC on validation set for reference
        preds_val = self.model.predict(
            dval, iteration_range=(0, self.model.best_iteration + 1)
        )
        preds_binary = (preds_val > 0.5).astype(int)
        mcc = matthews_corrcoef(y_val, preds_binary)
        logger.info(f"XGBoost Validation MCC (thresh=0.5): {mcc:.16f}")

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")

        # XGBoost requires DMatrix for prediction
        dtest = xgb.DMatrix(X)
        return self.model.predict(
            dtest, iteration_range=(0, self.model.best_iteration + 1)
        )
