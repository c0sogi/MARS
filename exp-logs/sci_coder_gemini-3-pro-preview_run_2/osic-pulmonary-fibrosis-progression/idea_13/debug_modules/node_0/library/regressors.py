import numpy as np
from sklearn.linear_model import QuantileRegressor, ElasticNet
from library.config import Config
from library.utils import seed_everything


class QuantileMedianRegressor:
    """
    A wrapper around sklearn.linear_model.QuantileRegressor configured to predict
    the median (50th percentile) of the target variable.

    This model minimizes the Mean Absolute Error (L1 loss), which is the optimal
    estimator for the location parameter of a Laplace distribution.
    """

    def __init__(self, alpha=0.01, **kwargs):
        """
        Initialize the QuantileMedianRegressor.

        Args:
            alpha (float): Regularization constant. Set to a small value (e.g., 0.01)
                           for stability, or 0.0 for pure Linear Quantile Regression.
            **kwargs: Additional arguments passed to QuantileRegressor.
        """
        seed_everything(Config.SEED)

        # We use the 'highs' solver which is generally the most efficient for
        # linear programming based quantile regression in recent sklearn versions.
        self.model = QuantileRegressor(
            quantile=0.5, alpha=alpha, solver="highs", **kwargs
        )

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Args:
            X (array-like): Feature matrix.
            y (array-like): Target vector (FVC).

        Returns:
            self
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict the median FVC.

        Args:
            X (array-like): Feature matrix.

        Returns:
            np.ndarray: Predicted median values.
        """
        return self.model.predict(X)


class ResidualUncertaintyRegressor:
    """
    A wrapper around sklearn.linear_model.ElasticNet configured to predict
    the magnitude of prediction errors (uncertainty).

    This model predicts the Mean Absolute Deviation (MAD) of the residuals
    from the median regressor.
    """

    def __init__(self, alpha=0.1, l1_ratio=0.5, **kwargs):
        """
        Initialize the ResidualUncertaintyRegressor.

        Args:
            alpha (float): Constant that multiplies the penalty terms.
            l1_ratio (float): The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
            **kwargs: Additional arguments passed to ElasticNet.
        """
        seed_everything(Config.SEED)

        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=Config.MAX_ITER,
            random_state=Config.SEED,
            **kwargs
        )

    def fit(self, X, residuals):
        """
        Fit the model to the absolute residuals.

        Args:
            X (array-like): Feature matrix.
            residuals (array-like): Raw residuals (y_true - y_pred).
                                    The model will train on abs(residuals).

        Returns:
            self
        """
        # We predict the magnitude of error, so we train on absolute residuals
        y_target = np.abs(residuals)
        self.model.fit(X, y_target)
        return self

    def predict(self, X):
        """
        Predict the expected Mean Absolute Deviation (MAD).

        Args:
            X (array-like): Feature matrix.

        Returns:
            np.ndarray: Predicted uncertainty values (guaranteed non-negative).
        """
        predictions = self.model.predict(X)
        # Uncertainty cannot be negative
        return np.maximum(predictions, 0.0)
