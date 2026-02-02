import os
import numpy as np
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from library.utils import setup_logger
from library.config import SEED


class ModelWrapper(BaseEstimator, ClassifierMixin):
    """
    Abstract base class/interface for model wrappers.
    Enforces a common API for the heterogeneous ensemble.
    """

    def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=50):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError


class LGBMClassifierWrapper(ModelWrapper):
    """
    Wrapper for LightGBM Classifier.
    Handles training with early stopping and persistence.
    """

    def __init__(self, params, name="lgbm"):
        """
        Initialize the LightGBM wrapper.

        Args:
            params (dict): Dictionary of hyperparameters.
            name (str): Identifier for logging.
        """
        self.params = params.copy()
        self.name = name
        self.model = None
        self.logger = setup_logger()

    def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=50):
        """
        Trains the LightGBM model.

        Args:
            X (pd.DataFrame or np.ndarray): Training features.
            y (pd.Series or np.ndarray): Training labels.
            X_val (pd.DataFrame or np.ndarray, optional): Validation features.
            y_val (pd.Series or np.ndarray, optional): Validation labels.
            early_stopping_rounds (int): Rounds for early stopping.
        """
        self.logger.info(f"Training {self.name} with params: {self.params}")

        # Initialize the sklearn API model
        self.model = lgb.LGBMClassifier(**self.params)

        eval_set = None
        callbacks = []

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            # LightGBM 4.x uses callbacks for early stopping
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
            callbacks.append(lgb.log_evaluation(period=100))

        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            eval_metric=self.params.get("metric", "auc"),
            callbacks=callbacks,
        )

        if hasattr(self.model, "best_score_"):
            self.logger.info(f"{self.name} Best Score: {self.model.best_score_}")

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X (pd.DataFrame or np.ndarray): Features.

        Returns:
            np.ndarray: Probability of the positive class (class 1).
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # predict_proba returns [prob_0, prob_1]
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        """
        Saves the model to disk using joblib.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        self.logger.info(f"Saved {self.name} model to {path}")

    def load(self, path):
        """
        Loads the model from disk.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        self.model = joblib.load(path)
        self.logger.info(f"Loaded {self.name} model from {path}")
        return self


class XGBClassifierWrapper(ModelWrapper):
    """
    Wrapper for XGBoost Classifier.
    Handles training with early stopping and persistence.
    """

    def __init__(self, params, name="xgb"):
        """
        Initialize the XGBoost wrapper.

        Args:
            params (dict): Dictionary of hyperparameters.
            name (str): Identifier for logging.
        """
        self.params = params.copy()
        self.name = name
        self.model = None
        self.logger = setup_logger()

    def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=50):
        """
        Trains the XGBoost model.

        Args:
            X (pd.DataFrame or np.ndarray): Training features.
            y (pd.Series or np.ndarray): Training labels.
            X_val (pd.DataFrame or np.ndarray, optional): Validation features.
            y_val (pd.Series or np.ndarray, optional): Validation labels.
            early_stopping_rounds (int): Rounds for early stopping.
        """
        self.logger.info(f"Training {self.name} with params: {self.params}")

        init_params = self.params.copy()
        eval_set = None

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            init_params["early_stopping_rounds"] = early_stopping_rounds

        self.model = xgb.XGBClassifier(**init_params)

        # XGBoost sklearn API handles early_stopping_rounds in constructor now
        # verbose=False to keep logs clean
        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            verbose=False,
        )

        if hasattr(self.model, "best_score"):
            self.logger.info(f"{self.name} Best Score: {self.model.best_score}")

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X (pd.DataFrame or np.ndarray): Features.

        Returns:
            np.ndarray: Probability of the positive class (class 1).
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        """
        Saves the model to disk using joblib.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        self.logger.info(f"Saved {self.name} model to {path}")

    def load(self, path):
        """
        Loads the model from disk.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        self.model = joblib.load(path)
        self.logger.info(f"Loaded {self.name} model from {path}")
        return self
