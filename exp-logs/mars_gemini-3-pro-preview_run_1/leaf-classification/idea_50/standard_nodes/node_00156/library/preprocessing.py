import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler


class RobustPipeline:
    """
    Encapsulates the Inductive Preprocessing Pipeline:
    1. Yeo-Johnson Power Transformation (standardize=False)
    2. Standard Scaling

    Ensures all operations are performed in float64 precision to maintain
    numerical stability for the OAS estimator.
    """

    def __init__(self):
        # Initialize transformers
        # We set standardize=False in PowerTransformer because we apply
        # a dedicated StandardScaler explicitly afterwards.
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X, y=None):
        """
        Fits the pipeline on the training data.

        Args:
            X (pd.DataFrame or np.ndarray): Training features.
            y (pd.Series or np.ndarray, optional): Target labels. Ignored.

        Returns:
            self
        """
        # Enforce float64 precision explicitly
        # This prevents hidden downcasting to float32 which could hurt the OAS inversion
        X_float = X.astype(np.float64)

        # Store feature names if available for reconstruction later
        if hasattr(X, "columns"):
            self.feature_names = X.columns

        # 1. Fit PowerTransformer (Yeo-Johnson)
        # This attempts to make the data more Gaussian-like
        self.pt.fit(X_float)

        # 2. Transform to get intermediate state for Scaler fitting
        X_pt = self.pt.transform(X_float)

        # 3. Fit StandardScaler
        # Centers and scales the Gaussianized data
        self.scaler.fit(X_pt)

        self.is_fitted = True
        return self

    def transform(self, X):
        """
        Applies the fitted transformations to the data.

        Args:
            X (pd.DataFrame or np.ndarray): Features to transform.

        Returns:
            pd.DataFrame or np.ndarray: Transformed features in float64.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "RobustPipeline must be fitted before calling transform."
            )

        # Enforce float64 precision
        X_float = X.astype(np.float64)

        # 1. Apply PowerTransformer
        X_pt = self.pt.transform(X_float)

        # 2. Apply StandardScaler
        X_scaled = self.scaler.transform(X_pt)

        # Reconstruct DataFrame if metadata is available or input was DataFrame
        # This ensures column names are preserved for feature importance analysis
        index = X.index if hasattr(X, "index") else None
        columns = (
            self.feature_names
            if self.feature_names is not None
            else (X.columns if hasattr(X, "columns") else None)
        )

        if columns is not None:
            return pd.DataFrame(X_scaled, columns=columns, index=index)

        return X_scaled

    def fit_transform(self, X, y=None):
        """
        Fits and transforms the data.

        Args:
            X (pd.DataFrame or np.ndarray): Training features.
            y (pd.Series or np.ndarray, optional): Target labels.

        Returns:
            pd.DataFrame or np.ndarray: Transformed features.
        """
        return self.fit(X, y).transform(X)
