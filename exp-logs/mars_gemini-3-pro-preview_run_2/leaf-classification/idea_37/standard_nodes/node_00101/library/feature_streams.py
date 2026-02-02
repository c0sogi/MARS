import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from library.config import Config


class DualStreamPreprocessor(BaseEstimator, TransformerMixin):
    """
    Implements the Dual-Stream preprocessing logic for the Constrained-Basis
    Dual-Stream Generative Ensemble.

    This transformer maintains two parallel preprocessing streams:
    1. Stream A: Parametric Gaussian Anchors (Yeo-Johnson)
    2. Stream B: Constrained Non-Parametric Experts (Quantile, n=50)

    All outputs are strictly cast to float64 to ensure precision near the metric floor.
    """

    def __init__(
        self,
        pt_method=Config.PT_METHOD,
        pt_standardize=Config.PT_STANDARDIZE,
        qt_output_dist=Config.QT_OUTPUT_DIST,
        qt_n_quantiles=Config.QT_N_QUANTILES,
        random_state=Config.RANDOM_STATE,
        dtype=Config.NP_DTYPE,
    ):
        """
        Args:
            pt_method (str): Method for PowerTransformer (default: 'yeo-johnson').
            pt_standardize (bool): Whether to standardize in PowerTransformer.
            qt_output_dist (str): Output distribution for QuantileTransformer (default: 'normal').
            qt_n_quantiles (int): Number of quantiles for QuantileTransformer (default: 50).
            random_state (int): Seed for reproducibility.
            dtype (numpy.dtype): Data type for output arrays (default: np.float64).
        """
        self.pt_method = pt_method
        self.pt_standardize = pt_standardize
        self.qt_output_dist = qt_output_dist
        self.qt_n_quantiles = qt_n_quantiles
        self.random_state = random_state
        self.dtype = dtype

        self.pt = None
        self.qt = None

    def fit(self, X, y=None):
        """
        Fits both the Parametric and Constrained Non-Parametric transformers on X.

        Args:
            X (array-like): Input data of shape (n_samples, n_features).
            y (ignored): Not used, present for API consistency.

        Returns:
            self: The fitted transformer.
        """
        # Initialize Stream A: Parametric
        self.pt = PowerTransformer(
            method=self.pt_method, standardize=self.pt_standardize
        )

        # Initialize Stream B: Constrained Non-Parametric
        self.qt = QuantileTransformer(
            output_distribution=self.qt_output_dist,
            n_quantiles=self.qt_n_quantiles,
            random_state=self.random_state,
        )

        # Fit both streams
        self.pt.fit(X)
        self.qt.fit(X)

        return self

    def transform(self, X):
        """
        Transforms X using both streams.

        Args:
            X (array-like): Input data of shape (n_samples, n_features).

        Returns:
            dict: A dictionary containing the transformed data:
                - 'stream_a': Output from PowerTransformer (float64).
                - 'stream_b': Output from QuantileTransformer (float64).
        """
        if self.pt is None or self.qt is None:
            raise RuntimeError(
                "This DualStreamPreprocessor instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

        # Transform Stream A
        xa = self.pt.transform(X).astype(self.dtype)

        # Transform Stream B
        xb = self.qt.transform(X).astype(self.dtype)

        return {"stream_a": xa, "stream_b": xb}
