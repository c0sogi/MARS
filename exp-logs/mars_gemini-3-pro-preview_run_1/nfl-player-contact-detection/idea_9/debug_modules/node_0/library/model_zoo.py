import os
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import joblib
from library.config import (
    WORKING_DIR,
    N_ESTIMATORS,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    SEED,
)
from library.utils import setup_logging


class LGBMWrapper:
    """
    Wrapper for LightGBM model training and inference.
    """

    def __init__(self, params, model_name="lgbm_model"):
        self.params = params
        self.model_name = model_name
        self.model = None
        self.logger = setup_logging()
        self.model_dir = os.path.join(WORKING_DIR, "models")
        os.makedirs(self.model_dir, exist_ok=True)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the LightGBM model with early stopping.
        """
        self.logger.info(f"Training LightGBM model: {self.model_name}")

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")

        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=VERBOSE_EVAL),
        ]

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=N_ESTIMATORS,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # Log best score
        if self.model.best_score:
            self.logger.info(f"Best Score: {self.model.best_score}")

    def predict(self, X):
        """
        Predicts probabilities.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def save(self):
        """
        Saves the model to disk.
        """
        if self.model is None:
            self.logger.warning("No model to save.")
            return

        path = os.path.join(self.model_dir, f"{self.model_name}.joblib")
        joblib.dump(self.model, path)
        self.logger.info(f"Saved LightGBM model to {path}")

    def load(self):
        """
        Loads the model from disk.
        """
        path = os.path.join(self.model_dir, f"{self.model_name}.joblib")
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.logger.info(f"Loaded LightGBM model from {path}")
            return True
        else:
            self.logger.warning(f"Model file not found at {path}")
            return False


class XGBWrapper:
    """
    Wrapper for XGBoost model training and inference.
    """

    def __init__(self, params, model_name="xgb_model"):
        self.params = params
        self.model_name = model_name
        self.model = None
        self.logger = setup_logging()
        self.model_dir = os.path.join(WORKING_DIR, "models")
        os.makedirs(self.model_dir, exist_ok=True)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the XGBoost model with early stopping.
        """
        self.logger.info(f"Training XGBoost model: {self.model_name}")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, "train")]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "valid"))

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=N_ESTIMATORS,
            evals=evals,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=VERBOSE_EVAL,
        )

    def predict(self, X):
        """
        Predicts probabilities.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")

        dtest = xgb.DMatrix(X)
        return self.model.predict(
            dtest, iteration_range=(0, self.model.best_iteration + 1)
        )

    def save(self):
        """
        Saves the model to disk.
        """
        if self.model is None:
            self.logger.warning("No model to save.")
            return

        path = os.path.join(self.model_dir, f"{self.model_name}.joblib")
        joblib.dump(self.model, path)
        self.logger.info(f"Saved XGBoost model to {path}")

    def load(self):
        """
        Loads the model from disk.
        """
        path = os.path.join(self.model_dir, f"{self.model_name}.joblib")
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.logger.info(f"Loaded XGBoost model from {path}")
            return True
        else:
            self.logger.warning(f"Model file not found at {path}")
            return False


class EnsembleModel:
    """
    Heterogeneous Ensemble of LightGBM and XGBoost.
    """

    def __init__(self, lgbm_params, xgb_params):
        self.lgbm = LGBMWrapper(lgbm_params, model_name="expert_lgbm")
        self.xgb = XGBWrapper(xgb_params, model_name="expert_xgb")
        self.logger = setup_logging()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains both models sequentially.
        """
        self.logger.info("Starting Ensemble Training...")

        # Train LightGBM
        self.lgbm.fit(X_train, y_train, X_val, y_val)
        self.lgbm.save()

        # Train XGBoost
        self.xgb.fit(X_train, y_train, X_val, y_val)
        self.xgb.save()

        self.logger.info("Ensemble Training Complete.")

    def predict(self, X):
        """
        Returns the unweighted average of probabilities from both models.
        """
        self.logger.info("Generating Ensemble Predictions...")

        pred_lgbm = self.lgbm.predict(X)
        pred_xgb = self.xgb.predict(X)

        # Unweighted Average
        ensemble_pred = (pred_lgbm + pred_xgb) / 2.0
        return ensemble_pred

    def load(self):
        """
        Loads both models from disk. Returns True if both loaded successfully.
        """
        lgbm_loaded = self.lgbm.load()
        xgb_loaded = self.xgb.load()
        return lgbm_loaded and xgb_loaded
