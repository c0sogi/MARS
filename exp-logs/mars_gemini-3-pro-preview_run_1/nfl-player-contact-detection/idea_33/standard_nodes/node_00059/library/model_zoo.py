import os
import joblib
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from library.config import Config


class BaseModel:
    """
    Abstract base class for all models in the DEIB-AME ensemble.
    Enforces a consistent API for training, inference, and persistence.
    """

    def __init__(self, name="base_model"):
        self.model = None
        self.name = name

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Trains the model.
        Args:
            X: Training features.
            y: Training targets.
            X_val: Validation features (optional, for early stopping).
            y_val: Validation targets (optional, for early stopping).
        """
        raise NotImplementedError

    def predict_proba(self, X):
        """
        Predicts class probabilities.
        Args:
            X: Features.
        Returns:
            np.ndarray: Probability of the positive class (contact).
        """
        raise NotImplementedError

    def predict(self, X):
        """
        Predicts class labels.
        Args:
            X: Features.
        Returns:
            np.ndarray: Binary class labels.
        """
        raise NotImplementedError

    def save(self, path):
        """
        Saves the model to disk using joblib.
        """
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        joblib.dump(self.model, path)
        # print(f"Model saved to {path}")

    def load(self, path):
        """
        Loads the model from disk using joblib.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(path)
        # print(f"Model loaded from {path}")


class LGBMExpert(BaseModel):
    """
    LightGBM Expert Model (Leaf-wise growth).
    Optimized for dense numerical data and handling imbalance internally.
    """

    def __init__(self):
        super().__init__(name="lgbm_expert")
        self.params = Config.LGBM_PARAMS.copy()
        self.model = lgb.LGBMClassifier(**self.params)

    def fit(self, X, y, X_val=None, y_val=None):
        callbacks = []
        eval_set = None

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            # Early stopping callback
            callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))
            # Logging callback (optional, keeping verbose=-1 in params usually suppresses this)
            callbacks.append(lgb.log_evaluation(period=100))

        self.model.fit(
            X, y, eval_set=eval_set, eval_metric="binary_logloss", callbacks=callbacks
        )

    def predict_proba(self, X):
        # Returns probability of class 1
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.model.predict(X)


class XGBExpert(BaseModel):
    """
    XGBoost Expert Model (Level-wise growth).
    """

    def __init__(self):
        super().__init__(name="xgb_expert")
        self.params = Config.XGB_PARAMS.copy()
        self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X, y, X_val=None, y_val=None):
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        # XGBoost handles early stopping via fit parameters
        # Note: early_stopping_rounds in fit() is the standard sklearn API approach
        self.model.fit(X, y, eval_set=eval_set, verbose=False)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.model.predict(X)


class HistGBExpert(BaseModel):
    """
    Histogram-based Gradient Boosting Expert (Scikit-Learn).
    Serves as the third diverse estimator (replacing CatBoost).
    Uses internal validation splitting for early stopping.
    """

    def __init__(self):
        super().__init__(name="hgb_expert")
        self.params = Config.HGB_PARAMS.copy()
        self.model = HistGradientBoostingClassifier(**self.params)

    def fit(self, X, y, X_val=None, y_val=None):
        # HistGradientBoostingClassifier uses 'validation_fraction' from params
        # to split X internally for early stopping if early_stopping=True.
        # It does not accept X_val explicitly in fit() for early stopping control
        # in the same way LGBM/XGB do.
        # We simply fit on the provided training data.
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.model.predict(X)
