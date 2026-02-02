import os
import joblib
import lightgbm as lgb
import xgboost as xgb
from abc import ABC, abstractmethod


class ModelInterface(ABC):
    """
    Abstract base class for model wrappers ensuring a consistent API
    for training, prediction, and persistence across different algorithms.
    """

    @abstractmethod
    def fit(self, X, y, eval_set=None):
        """
        Trains the model with optional validation data for early stopping.

        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix.
            y (pd.Series or np.ndarray): Target vector.
            eval_set (list): Optional list of (X, y) tuples for validation.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predicts class probabilities for the positive class (contact).

        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Probabilities of class 1.
        """
        pass

    @abstractmethod
    def save(self, path):
        """Saves the model to disk."""
        pass

    @abstractmethod
    def load(self, path):
        """Loads the model from disk."""
        pass


class LGBMWrapper(ModelInterface):
    """
    Wrapper for LightGBM Classifier.
    Handles LightGBM-specific callbacks for early stopping and silent execution.
    """

    def __init__(self, params):
        self.params = params.copy()
        self.model = lgb.LGBMClassifier(**self.params)

    def fit(self, X, y, eval_set=None):
        callbacks = []
        if eval_set:
            # Use callbacks for early stopping in recent LightGBM versions
            # stopping_rounds=50 as per strategy
            callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))
            # Suppress evaluation logging
            callbacks.append(lgb.log_evaluation(period=0))

        self.model.fit(X, y, eval_set=eval_set, eval_metric="auc", callbacks=callbacks)

    def predict(self, X):
        # Return probabilities for the positive class (contact)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path):
        if os.path.exists(path):
            self.model = joblib.load(path)
        else:
            raise FileNotFoundError(f"Model file not found at {path}")


class XGBWrapper(ModelInterface):
    """
    Wrapper for XGBoost Classifier.
    Handles XGBoost-specific arguments for early stopping and silent execution.
    """

    def __init__(self, params):
        self.params = params.copy()
        self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X, y, eval_set=None):
        # XGBoost >= 1.6.0 requires early_stopping_rounds to be set via init or set_params
        # Cite debug_lesson_7
        if eval_set:
            self.model.set_params(early_stopping_rounds=50)
        else:
            self.model.set_params(early_stopping_rounds=None)

        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            verbose=False,
        )

    def predict(self, X):
        # Return probabilities for the positive class (contact)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path):
        if os.path.exists(path):
            self.model = joblib.load(path)
        else:
            raise FileNotFoundError(f"Model file not found at {path}")
