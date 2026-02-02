import numpy as np
from sklearn.linear_model import QuantileRegressor, ElasticNet
from library.config import Config


class FVCPredictor:
    """
    Wrapper for the Median FVC Predictor using Quantile Regression.

    This model predicts the 50th percentile (median) of the FVC distribution.
    Predicting the median minimizes the Mean Absolute Error (L1 loss), which
    aligns with the location parameter optimization for the Laplace Log Likelihood metric.
    """

    def __init__(self):
        # Initialize QuantileRegressor
        # quantile=0.5: Target the median.
        # alpha=0.0: Use standard Quantile Regression (Least Absolute Deviations)
        #            without additional L1 regularization on the coefficients.
        # solver='highs': A robust linear programming solver suitable for this dataset size.
        # solver_options: Passes execution limits to the solver.
        self.model = QuantileRegressor(
            quantile=Config.QUANTILE,
            alpha=0.0,
            solver="highs",
            solver_options={"max_iter": Config.QR_MAX_ITER},
        )

    def fit(self, X, y):
        """
        Fits the Quantile Regressor to the training data.

        Args:
            X (np.ndarray): Feature matrix (typically Static + Time + Interactions).
            y (np.ndarray): Target FVC values.

        Returns:
            self: The fitted predictor instance.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts the median FVC for the given inputs.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted median FVC values.
        """
        return self.model.predict(X)


class UncertaintyPredictor:
    """
    Wrapper for the Uncertainty Predictor using Elastic Net.

    This model predicts the magnitude of the error (absolute residual) expected
    for a given patient. This serves as a proxy for the scale parameter (sigma)
    in the Laplace distribution.
    """

    def __init__(self):
        # Initialize ElasticNet
        # Uses a mix of L1 (Lasso) and L2 (Ridge) regularization to robustly
        # estimate the conditional mean of the absolute errors.
        self.model = ElasticNet(
            alpha=Config.EN_ALPHA,
            l1_ratio=Config.EN_L1_RATIO,
            max_iter=Config.EN_MAX_ITER,
            random_state=Config.SEED,
        )

    def fit(self, X, y):
        """
        Fits the Elastic Net model to the absolute residuals.

        Args:
            X (np.ndarray): Static feature matrix (typically excluding time interactions).
            y (np.ndarray): Absolute residuals (|y_true - y_pred_median|).

        Returns:
            self: The fitted predictor instance.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts the expected Mean Absolute Deviation (MAD).

        Args:
            X (np.ndarray): Static feature matrix.

        Returns:
            np.ndarray: Predicted MAD values.
        """
        return self.model.predict(X)
