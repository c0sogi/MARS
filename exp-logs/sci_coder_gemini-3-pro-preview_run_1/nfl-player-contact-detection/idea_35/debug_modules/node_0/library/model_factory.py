import os
import joblib
import numpy as np
import logging
from abc import ABC, abstractmethod
from library.config import Config
from library.utils import setup_logger

# Initialize module logger
logger = setup_logger("model_factory")

# Import Gradient Boosting Libraries
try:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    logger.warning("LightGBM not found. LGBMExpert will fail if initialized.")

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost not found. XGBExpert will fail if initialized.")

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    # CatBoost is required by the architecture but might be missing in the env list.
    # We warn but allow the class definition to exist.
    logger.warning("CatBoost not found. CatBoostExpert will fail if initialized.")


class ModelWrapper(ABC):
    """
    Abstract Base Class for Expert Models in the Tri-Ensemble.
    Enforces a consistent API for fit, predict, save, and load.
    """

    def __init__(self, params: dict):
        self.params = params.copy()
        self.model = None

    @abstractmethod
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Trains the model with optional validation data for early stopping.
        """
        pass

    def predict_proba(self, X):
        """
        Predicts class probabilities.
        Returns:
            np.ndarray: Probability of the positive class (contact).
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Most sklearn-API classifiers return shape (N, 2)
        # We want the probability of class 1
        probs = self.model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        return probs

    def save(self, path: str):
        """Saves the model to disk."""
        if self.model is None:
            raise ValueError("Cannot save an untrained model.")
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Loads the model from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")


class LGBMExpert(ModelWrapper):
    """
    LightGBM Expert implementation (Leaf-wise growth).
    Optimized for dense numerical data.
    """

    def __init__(self):
        if not HAS_LGBM:
            raise ImportError("LightGBM is not installed.")
        super().__init__(Config.LGBM_PARAMS)
        self.model = LGBMClassifier(**self.params)

    def fit(self, X, y, X_val=None, y_val=None):
        logger.info("Training LGBMExpert...")

        callbacks = []
        eval_set = None

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            # Configure callbacks for early stopping and logging
            callbacks.append(
                early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS)
            )
            callbacks.append(log_evaluation(period=Config.VERBOSE_EVAL))

        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            eval_metric=self.params.get("metric", "binary_logloss"),
            callbacks=callbacks,
        )

        # Log validation score if available
        if eval_set:
            best_score = self.model.best_score_["valid_0"][
                self.params.get("metric", "binary_logloss")
            ]
            logger.info(f"LGBMExpert Best Validation Score: {best_score}")


class XGBExpert(ModelWrapper):
    """
    XGBoost Expert implementation (Level-wise growth).
    Optimized for approximate splits and stability.
    """

    def __init__(self):
        if not HAS_XGB:
            raise ImportError("XGBoost is not installed.")
        super().__init__(Config.XGB_PARAMS)
        # XGBoost 3.x prefers early_stopping_rounds in constructor or fit.
        # We pass params to constructor.
        self.model = XGBClassifier(**self.params)

    def fit(self, X, y, X_val=None, y_val=None):
        logger.info("Training XGBExpert...")

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        # Note: In newer XGBoost/Scikit-Learn APIs, early_stopping_rounds is passed to fit
        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            verbose=Config.VERBOSE_EVAL,
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS if eval_set else None,
        )

        if hasattr(self.model, "best_score"):
            logger.info(f"XGBExpert Best Validation Score: {self.model.best_score}")


class CatBoostExpert(ModelWrapper):
    """
    CatBoost Expert implementation (Symmetric trees).
    Optimized for categorical handling and reducing overfitting.
    """

    def __init__(self):
        if not HAS_CATBOOST:
            raise ImportError("CatBoost is not installed.")
        super().__init__(Config.CATBOOST_PARAMS)
        self.model = CatBoostClassifier(**self.params)

    def fit(self, X, y, X_val=None, y_val=None):
        logger.info("Training CatBoostExpert...")

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)

        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS if eval_set else None,
            verbose=Config.VERBOSE_EVAL,
        )

        if hasattr(self.model, "best_score_"):
            # CatBoost stores best score in a dict structure
            logger.info(f"CatBoostExpert Best Score: {self.model.best_score_}")


def get_model(model_name: str) -> ModelWrapper:
    """
    Factory method to retrieve an expert model instance by name.

    Args:
        model_name (str): One of 'lgbm', 'xgb', 'catboost'.

    Returns:
        ModelWrapper: An instance of the requested expert.
    """
    if model_name.lower() == "lgbm":
        return LGBMExpert()
    elif model_name.lower() == "xgb":
        return XGBExpert()
    elif model_name.lower() == "catboost":
        return CatBoostExpert()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
