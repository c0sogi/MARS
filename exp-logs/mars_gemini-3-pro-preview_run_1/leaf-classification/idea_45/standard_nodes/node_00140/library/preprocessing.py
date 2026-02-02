import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
import library.config as config


class HighPrecisionPreprocessor:
    """
    Implements the inductive preprocessing pipeline for the High-Precision OAS Discriminant.

    Pipeline Steps:
    1. Yeo-Johnson Power Transformation (stabilizes variance, standardize=False).
    2. Standard Scaling (centers and scales).

    Crucially ensures all input and output data remains in float64 precision to
    prevent precision loss before the linear solver.
    """

    def __init__(self, power_method=config.PREPROCESS_POWER_METHOD):
        """
        Initialize the preprocessor.

        Args:
            power_method (str): The method for PowerTransformer (default: 'yeo-johnson').
        """
        # Yeo-Johnson stabilizes variance for geometric features.
        # We set standardize=False because we apply StandardScaler explicitly afterwards.
        self.pt = PowerTransformer(method=power_method, standardize=False)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X):
        """
        Fits the inductive pipeline on the training data.

        Args:
            X (np.ndarray): Training data features.

        Returns:
            self: The fitted instance.
        """
        # Enforce float64 precision
        X_64 = X.astype(config.FLOAT_PRECISION)

        # 1. Fit PowerTransformer
        self.pt.fit(X_64)

        # 2. Transform data to intermediate space to fit Scaler
        # We must transform because Scaler needs to see the distribution
        # *after* power transformation.
        X_pt = self.pt.transform(X_64)

        # 3. Fit StandardScaler
        self.scaler.fit(X_pt)

        self.is_fitted = True
        return self

    def transform(self, X):
        """
        Applies the fitted transformations to the data.

        Args:
            X (np.ndarray): Data to transform.

        Returns:
            np.ndarray: Transformed data in float64.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "HighPrecisionPreprocessor must be fitted before calling transform."
            )

        # Enforce float64 precision
        X_64 = X.astype(config.FLOAT_PRECISION)

        # 1. Apply PowerTransformer
        X_pt = self.pt.transform(X_64)

        # 2. Apply StandardScaler
        X_final = self.scaler.transform(X_pt)

        # Ensure output is explicitly float64
        return X_final.astype(config.FLOAT_PRECISION)

    def fit_transform(self, X):
        """
        Fits the preprocessor on X and returns the transformed version.

        Args:
            X (np.ndarray): Training data features.

        Returns:
            np.ndarray: Transformed training data in float64.
        """
        self.fit(X)
        return self.transform(X)
