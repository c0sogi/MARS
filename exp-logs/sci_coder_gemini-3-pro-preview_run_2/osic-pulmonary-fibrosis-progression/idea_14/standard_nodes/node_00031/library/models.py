import os
import numpy as np
import joblib
from sklearn.linear_model import QuantileRegressor, ElasticNet
from library.config import Config


class QuantileLinearModel:
    """
    A wrapper around sklearn's QuantileRegressor for FVC prediction.
    Target: Median FVC (quantile=0.5).
    Optimization: Minimizes L1 loss (Mean Absolute Error).
    """

    def __init__(self, quantile=None, alpha=0.0):
        """
        Args:
            quantile (float): The quantile to predict. Defaults to Config.QUANTILE (0.5).
            alpha (float): Regularization strength. Defaults to 0.0 (Pure Quantile Regression).
                           Set > 0 for L1 regularization on coefficients.
        """
        self.quantile = quantile if quantile is not None else Config.QUANTILE
        self.alpha = alpha

        # Initialize the regressor
        # solver='highs' is efficient for linear programming problems in recent sklearn versions
        self.model = QuantileRegressor(
            quantile=self.quantile, alpha=self.alpha, solver="highs"
        )

    def fit(self, X, y):
        """
        Fits the quantile regression model.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts the conditional median FVC.
        """
        return self.model.predict(X)

    def save(self, filename):
        """
        Saves the trained model to the cache directory.
        """
        path = os.path.join(Config.CACHE_DIR, filename)
        joblib.dump(self.model, path)

    def load(self, filename):
        """
        Loads a trained model from the cache directory.
        """
        path = os.path.join(Config.CACHE_DIR, filename)
        if os.path.exists(path):
            self.model = joblib.load(path)
        else:
            raise FileNotFoundError(f"Model file not found: {path}")


class ResidualElasticModel:
    """
    A wrapper around sklearn's ElasticNet for Uncertainty prediction.
    Target: Absolute Residuals (|y_true - y_pred|).
    Optimization: Minimizes L2 loss with L1+L2 regularization.
    """

    def __init__(self, alpha=None, l1_ratio=None):
        """
        Args:
            alpha (float): Constant that multiplies the penalty terms. Defaults to Config.
            l1_ratio (float): The mixing parameter, with 0 <= l1_ratio <= 1. Defaults to Config.
        """
        self.alpha = alpha if alpha is not None else Config.ELASTIC_NET_ALPHA
        self.l1_ratio = (
            l1_ratio if l1_ratio is not None else Config.ELASTIC_NET_L1_RATIO
        )

        # Initialize ElasticNet
        # Sets random_state for reproducibility
        self.model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            random_state=Config.SEED,
            max_iter=5000,  # Ensure convergence
        )

    def fit(self, X, y):
        """
        Fits the ElasticNet model on the residuals.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts the expected Mean Absolute Deviation (MAD).
        Note: The output should be converted to sigma (std dev) using sigma = MAD * sqrt(2)
        in the evaluation/inference pipeline, not here.
        """
        pred = self.model.predict(X)
        # Uncertainty (magnitude of error) cannot be negative.
        return np.maximum(pred, 0)

    def save(self, filename):
        """
        Saves the trained model to the cache directory.
        """
        path = os.path.join(Config.CACHE_DIR, filename)
        joblib.dump(self.model, path)

    def load(self, filename):
        """
        Loads a trained model from the cache directory.
        """
        path = os.path.join(Config.CACHE_DIR, filename)
        if os.path.exists(path):
            self.model = joblib.load(path)
        else:
            raise FileNotFoundError(f"Model file not found: {path}")
