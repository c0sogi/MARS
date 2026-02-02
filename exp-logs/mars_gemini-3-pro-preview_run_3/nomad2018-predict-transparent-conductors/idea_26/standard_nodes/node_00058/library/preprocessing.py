import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer


class TargetTransformer:
    """
    Handles log(1+y) transformation for targets and inverse transformation for predictions.
    This stabilizes variance for regression tasks on energy values which are strictly positive
    and can span multiple orders of magnitude.
    """

    def __init__(self):
        pass

    def transform(self, y):
        """
        Apply log(1+y) transformation.
        Args:
            y (np.array or pd.Series): Target values.
        Returns:
            np.array: Log-transformed values.
        """
        # Ensure input is numpy array
        y_arr = np.array(y)
        # Clip negative values to 0 before log1p to avoid NaNs, though energies should be >= 0
        return np.log1p(np.maximum(y_arr, 0))

    def inverse_transform(self, z):
        """
        Apply exp(z) - 1 transformation.
        Args:
            z (np.array or pd.Series): Log-transformed predictions.
        Returns:
            np.array: Original scale predictions.
        """
        z_arr = np.array(z)
        # Apply inverse
        y_pred = np.expm1(z_arr)
        # Ensure non-negative predictions as energies cannot be negative
        return np.maximum(y_pred, 0)


class FeatureCleaner(BaseEstimator, TransformerMixin):
    """
    Pipeline component to clean feature sets.
    1. Imputes missing values (NaNs) using the mean strategy.
    2. Removes constant (zero variance) features to prevent feature dilution.
    """

    def __init__(self, constant_threshold=0.0):
        """
        Args:
            constant_threshold (float): Threshold for variance. Features with variance <= this
                                        will be removed. Default is 0.0 (remove constant features).
        """
        self.constant_threshold = constant_threshold
        self.imputer = SimpleImputer(strategy="mean")
        self.selector = VarianceThreshold(threshold=constant_threshold)
        self.feature_names_in_ = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        """
        Fit the imputer and variance threshold selector.
        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix.
            y: Ignored.
        """
        # Store feature names if available
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"feat_{i}" for i in range(X.shape[1])]

        # Fit imputer
        self.imputer.fit(X)

        # Fit selector on imputed data (VarianceThreshold requires no NaNs)
        X_imputed = self.imputer.transform(X)
        self.selector.fit(X_imputed)

        # Determine output feature names based on support
        support = self.selector.get_support()
        self.feature_names_out_ = [
            name for name, keep in zip(self.feature_names_in_, support) if keep
        ]

        return self

    def transform(self, X):
        """
        Impute missing values and remove constant features.
        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix.
        Returns:
            pd.DataFrame or np.ndarray: Cleaned feature matrix.
        """
        # Impute
        X_imputed = self.imputer.transform(X)

        # Select features
        X_selected = self.selector.transform(X_imputed)

        # Return DataFrame if feature names are tracked, preserving index
        if self.feature_names_out_ is not None:
            index = X.index if hasattr(X, "index") else None
            return pd.DataFrame(
                X_selected, columns=self.feature_names_out_, index=index
            )

        return X_selected
