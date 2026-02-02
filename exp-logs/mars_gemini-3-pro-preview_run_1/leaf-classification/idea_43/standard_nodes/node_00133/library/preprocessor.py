import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import PRECISION_TYPE


class HighPrecisionPreprocessor:
    """
    Implements the Inductive Preprocessing Pipeline with strict float64 precision.

    This class wraps sklearn's PowerTransformer (Yeo-Johnson) and StandardScaler
    to stabilize feature variance and normalize distributions. It is designed to
    be fitted ONLY on the training set and then applied to train, validation,
    and test sets to prevent data leakage.
    """

    def __init__(self, use_yeo_johnson=True, standardize=True):
        """
        Args:
            use_yeo_johnson (bool): Whether to apply Yeo-Johnson Power Transformation.
            standardize (bool): Whether to apply Standard Scaling (z-score).
        """
        self.use_yeo_johnson = use_yeo_johnson
        self.standardize = standardize

        # Transformers
        self.pt = None
        self.scaler = None

    def fit(self, X):
        """
        Fits the transformers on the provided data (Training set).

        Args:
            X (array-like): Training features.

        Returns:
            self
        """
        # Enforce strict double precision
        X_curr = np.array(X, dtype=PRECISION_TYPE)

        # 1. Fit Power Transformer (Yeo-Johnson)
        if self.use_yeo_johnson:
            # We set standardize=False because we want to handle scaling explicitly
            # in the next step.
            self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
            self.pt.fit(X_curr)

            # Transform the data so the next estimator (Scaler) sees the
            # Gaussian-like distribution
            X_curr = self.pt.transform(X_curr)

        # 2. Fit Standard Scaler
        if self.standardize:
            self.scaler = StandardScaler()
            self.scaler.fit(X_curr)

        return self

    def transform(self, X):
        """
        Applies the learned transformations to the data.

        Args:
            X (array-like): Features to transform (Train, Val, or Test).

        Returns:
            np.ndarray: Transformed features in float64.
        """
        # Enforce strict double precision
        X_curr = np.array(X, dtype=PRECISION_TYPE)

        # 1. Apply Power Transformer
        if self.use_yeo_johnson:
            if self.pt is None:
                raise RuntimeError(
                    "Preprocessor (Yeo-Johnson) must be fitted before transform."
                )
            X_curr = self.pt.transform(X_curr)

        # 2. Apply Standard Scaler
        if self.standardize:
            if self.scaler is None:
                raise RuntimeError(
                    "Preprocessor (StandardScaler) must be fitted before transform."
                )
            X_curr = self.scaler.transform(X_curr)

        return X_curr.astype(PRECISION_TYPE)

    def fit_transform(self, X):
        """
        Fits on X and returns the transformed X.

        Args:
            X (array-like): Training features.

        Returns:
            np.ndarray: Transformed features in float64.
        """
        self.fit(X)
        return self.transform(X)
