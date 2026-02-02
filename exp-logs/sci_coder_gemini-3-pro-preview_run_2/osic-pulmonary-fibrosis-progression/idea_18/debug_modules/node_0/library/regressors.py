import numpy as np
from sklearn.linear_model import QuantileRegressor, ElasticNet
from library.config import Config


def laplace_log_likelihood(y_true, y_pred, sigma):
    """
    Computes the modified Laplace Log Likelihood metric.

    Args:
        y_true (np.ndarray): True FVC values.
        y_pred (np.ndarray): Predicted FVC values.
        sigma (np.ndarray): Predicted confidence (std dev).

    Returns:
        float: The average metric score.
    """
    # Clip sigma to a minimum of 70
    sigma_clipped = np.maximum(sigma, 70)

    # Calculate absolute error
    delta = np.abs(y_true - y_pred)

    # Clip error at 1000
    delta_clipped = np.minimum(delta, 1000)

    # Calculate metric for each sample
    # metric = - (sqrt(2) * delta / sigma) - ln(sqrt(2) * sigma)
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta_clipped) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric)


class QuantileModel:
    """
    Wrapper for Linear Quantile Regression.
    Target: Median (q=0.5) to minimize L1 loss.
    """

    def __init__(self, quantile=0.5, alpha=0.0, max_iter=2000):
        """
        Args:
            quantile (float): Target quantile (default 0.5 for median).
            alpha (float): Regularization strength (0.0 for unregularized).
            max_iter (int): Maximum number of iterations for the solver.
        """
        # solver='highs' is generally faster for this scale of data
        self.model = QuantileRegressor(
            quantile=quantile,
            alpha=alpha,
            solver="highs",
            solver_options={"max_iter": max_iter},
        )

    def fit(self, X, y):
        """
        Fits the quantile regressor.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predicts the target quantile.
        """
        return self.model.predict(X)


class ResidualModel:
    """
    Wrapper for ElasticNet Regression to predict uncertainty.
    Target: Absolute Residuals (|y_true - y_pred|).
    """

    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=2000):
        """
        Args:
            alpha (float): Constant that multiplies the penalty terms.
            l1_ratio (float): The ElasticNet mixing parameter (0 <= l1_ratio <= 1).
            max_iter (int): Maximum number of iterations.
        """
        self.model = ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=Config.SEED
        )

    def fit(self, X, y):
        """
        Fits the ElasticNet model.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predicts the expected absolute residual.
        Enforces non-negativity as residuals cannot be negative.
        """
        preds = self.model.predict(X)
        return np.maximum(preds, 0)
