import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.covariance import OAS

from library.config import FLOAT_PRECISION, RANDOM_SEED


class BaseExpert(BaseEstimator, ClassifierMixin):
    """
    Base class for all experts in the ensemble.
    Enforces strict float64 precision and common interface.
    """

    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.model = None

    def fit(self, X, y):
        """
        Fit the underlying model.
        Args:
            X: Feature matrix (will be cast to FLOAT_PRECISION)
            y: Target vector
        """
        raise NotImplementedError

    def predict_proba(self, X):
        """
        Predict class probabilities.
        Args:
            X: Feature matrix (will be cast to FLOAT_PRECISION)
        Returns:
            Probability matrix of shape (n_samples, n_classes)
        """
        raise NotImplementedError


class LDAExpert(BaseExpert):
    """
    Generative Expert using Linear Discriminant Analysis.
    Supports fixed shrinkage and OAS covariance estimation.
    """

    def __init__(self, name, params):
        super().__init__(name, params)
        self._build_model()

    def _build_model(self):
        # Copy params to avoid modifying the original config
        model_params = self.params.copy()

        # Handle Covariance Estimator
        cov_est = model_params.get("covariance_estimator")
        if cov_est == "oas":
            # Instantiate OAS estimator
            model_params["covariance_estimator"] = OAS()
        elif cov_est is not None:
            # If it's not 'oas' but present, leave it (or handle other types if needed)
            # For this task, only 'oas' is specified in config for special handling.
            pass

        self.model = LinearDiscriminantAnalysis(**model_params)

    def fit(self, X, y):
        X_f64 = np.array(X, dtype=FLOAT_PRECISION)
        self.model.fit(X_f64, y)
        return self

    def predict_proba(self, X):
        X_f64 = np.array(X, dtype=FLOAT_PRECISION)
        return self.model.predict_proba(X_f64).astype(FLOAT_PRECISION)


class LogRegExpert(BaseExpert):
    """
    Discriminative Expert using Logistic Regression with Cross-Validation.
    Optimizes neg_log_loss over a dense grid of C values.
    """

    def __init__(self, name, params):
        super().__init__(name, params)
        self._build_model()

    def _build_model(self):
        model_params = self.params.copy()

        # Inject random state for reproducibility
        model_params["random_state"] = RANDOM_SEED

        # Ensure n_jobs is set if not present (though config usually has it)
        if "n_jobs" not in model_params:
            model_params["n_jobs"] = -1

        self.model = LogisticRegressionCV(**model_params)

    def fit(self, X, y):
        X_f64 = np.array(X, dtype=FLOAT_PRECISION)
        self.model.fit(X_f64, y)
        return self

    def predict_proba(self, X):
        X_f64 = np.array(X, dtype=FLOAT_PRECISION)
        return self.model.predict_proba(X_f64).astype(FLOAT_PRECISION)


def get_expert(config):
    """
    Factory function to instantiate the correct expert based on configuration.

    Args:
        config (dict): A dictionary containing 'name', 'type', and 'params'.
                       Example: {'name': 'lda_oas', 'type': 'lda', 'params': {...}}

    Returns:
        BaseExpert: An instance of LDAExpert or LogRegExpert.
    """
    expert_type = config.get("type")
    name = config.get("name")
    params = config.get("params", {})

    if expert_type == "lda":
        return LDAExpert(name, params)
    elif expert_type == "logreg_cv":
        return LogRegExpert(name, params)
    else:
        raise ValueError(f"Unknown expert type: {expert_type}")
