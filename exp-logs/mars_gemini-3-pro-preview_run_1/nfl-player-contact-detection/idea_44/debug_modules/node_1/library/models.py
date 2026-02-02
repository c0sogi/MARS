import os
import numpy as np
import joblib
import lightgbm as lgb
import xgboost as xgb
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

from library.config import Config
from library.utils import setup_logging


class ModelWrapper(ABC):
    """
    Abstract Base Class for model wrappers.
    Enforces a consistent interface for training, prediction, and persistence.
    """

    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params.copy()
        self.model = None
        self.logger = setup_logging()

    @abstractmethod
    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
    ):
        """
        Trains the model.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predicts probabilities for the positive class.
        """
        pass

    def save(self, output_dir: str):
        """
        Saves the model to the specified directory using joblib.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{self.name}.joblib")
        self.logger.info(f"Saving {self.name} model to {file_path}")
        joblib.dump(self.model, file_path)

    def load(self, input_dir: str):
        """
        Loads the model from the specified directory.
        """
        file_path = os.path.join(input_dir, f"{self.name}.joblib")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")

        self.logger.info(f"Loading {self.name} model from {file_path}")
        self.model = joblib.load(file_path)


class LGBMWrapper(ModelWrapper):
    """
    Wrapper for LightGBM Classifier using the Scikit-Learn API.
    """

    def __init__(self, name: str = "lgbm", params: Optional[Dict[str, Any]] = None):
        if params is None:
            params = Config.LGBM_PARAMS
        super().__init__(name, params)

        # Separate fit params from init params
        self.fit_params = {}
        if "early_stopping_rounds" in self.params:
            self.fit_params["early_stopping_rounds"] = self.params.pop(
                "early_stopping_rounds"
            )

        # Ensure verbosity is handled in init params
        if "verbose" not in self.params:
            self.params["verbose"] = -1

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
    ):
        self.logger.info(f"Training {self.name}...")

        # Initialize model
        self.model = lgb.LGBMClassifier(**self.params)

        callbacks = []
        eval_set = None

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            # Add early stopping callback if configured
            if "early_stopping_rounds" in self.fit_params:
                callbacks.append(
                    lgb.early_stopping(
                        stopping_rounds=self.fit_params["early_stopping_rounds"],
                        verbose=False,
                    )
                )
            # Add log evaluation callback to suppress output if verbose is -1 or False
            callbacks.append(lgb.log_evaluation(period=0))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_metric=self.params.get("metric", "binary_logloss"),
            callbacks=callbacks,
        )

        # Log best score if available
        if self.model.best_score_:
            # Accessing the metric name dynamically
            metric_name = list(self.model.best_score_["valid_0"].keys())[0]
            score = self.model.best_score_["valid_0"][metric_name]
            self.logger.info(f"{self.name} Best Validation {metric_name}: {score}")

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        # Return probability of class 1
        return self.model.predict_proba(X)[:, 1]


class XGBWrapper(ModelWrapper):
    """
    Wrapper for XGBoost Classifier using the Scikit-Learn API.
    """

    def __init__(self, name: str = "xgb", params: Optional[Dict[str, Any]] = None):
        if params is None:
            params = Config.XGB_PARAMS
        super().__init__(name, params)

        # Separate fit params from init params
        self.fit_params = {}
        if "early_stopping_rounds" in self.params:
            self.fit_params["early_stopping_rounds"] = self.params.pop(
                "early_stopping_rounds"
            )

        # Ensure verbosity is handled
        # XGBClassifier uses 'verbosity' (0=silent, 1=warning, 2=info, 3=debug)
        self.params["verbosity"] = 0

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
    ):
        self.logger.info(f"Training {self.name}...")

        self.model = xgb.XGBClassifier(**self.params)

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,  # Suppress per-iteration logging
            **self.fit_params,  # Passes early_stopping_rounds
        )

        if hasattr(self.model, "best_score"):
            self.logger.info(
                f"{self.name} Best Validation Score: {self.model.best_score}"
            )

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        return self.model.predict_proba(X)[:, 1]
