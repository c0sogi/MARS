import os
import joblib
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from library.config import Config
from library.utils import setup_logger


class LGBMClassifierWrapper:
    """
    Wrapper for LightGBM Classifier to standardize interface and handle
    specific training requirements like callbacks and early stopping.
    """

    def __init__(self, params):
        self.params = params.copy()
        self.model = None
        self.logger = setup_logger("LGBMWrapper")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the LightGBM model with optional early stopping.
        """
        # Initialize model with parameters
        self.model = lgb.LGBMClassifier(**self.params)

        callbacks = []
        # Configure Early Stopping via callbacks
        if Config.EARLY_STOPPING_ROUNDS > 0:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=False
                )
            )
            callbacks.append(lgb.log_evaluation(period=Config.VERBOSE_EVAL))

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_metric=self.params.get("metric", "auc"),
            callbacks=callbacks,
        )

    def predict_proba(self, X):
        """
        Returns probability estimates.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict_proba(X)

    def save(self, filepath):
        """
        Saves the model using joblib.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        """
        Loads the model using joblib.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        self.model = joblib.load(filepath)


class XGBClassifierWrapper:
    """
    Wrapper for XGBoost Classifier to standardize interface and handle
    version-specific early stopping configurations.
    """

    def __init__(self, params):
        self.params = params.copy()
        self.model = None
        self.logger = setup_logger("XGBWrapper")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the XGBoost model with optional early stopping.
        """
        fit_params = self.params.copy()
        eval_set = None

        # Configure Early Stopping via constructor parameters (XGBoost >= 1.6 style)
        if X_val is not None and y_val is not None:
            if Config.EARLY_STOPPING_ROUNDS > 0:
                fit_params["early_stopping_rounds"] = Config.EARLY_STOPPING_ROUNDS
            eval_set = [(X_val, y_val)]

        self.model = xgb.XGBClassifier(**fit_params)

        verbose_eval = False
        if Config.VERBOSE_EVAL > 0:
            verbose_eval = Config.VERBOSE_EVAL

        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose_eval)

    def predict_proba(self, X):
        """
        Returns probability estimates.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict_proba(X)

    def save(self, filepath):
        """
        Saves the model using joblib.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        """
        Loads the model using joblib.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        self.model = joblib.load(filepath)


class ModelFactory:
    """
    Factory class to instantiate models based on stage and type.
    """

    @staticmethod
    def create_model(stage: str, model_type: str):
        """
        Creates and returns a model wrapper instance.

        Args:
            stage (str): 'scout' or 'expert'.
            model_type (str): 'lgbm' or 'xgb'.

        Returns:
            Instance of LGBMClassifierWrapper or XGBClassifierWrapper.
        """
        if stage == "scout":
            if model_type == "lgbm":
                return LGBMClassifierWrapper(Config.SCOUT_LGBM_PARAMS)
            else:
                raise ValueError(
                    f"Scout stage only supports 'lgbm', got '{model_type}'"
                )

        elif stage == "expert":
            if model_type == "lgbm":
                return LGBMClassifierWrapper(Config.EXPERT_LGBM_PARAMS)
            elif model_type == "xgb":
                return XGBClassifierWrapper(Config.EXPERT_XGB_PARAMS)
            else:
                raise ValueError(
                    f"Expert stage supports 'lgbm' or 'xgb', got '{model_type}'"
                )

        else:
            raise ValueError(f"Unknown stage: {stage}")
