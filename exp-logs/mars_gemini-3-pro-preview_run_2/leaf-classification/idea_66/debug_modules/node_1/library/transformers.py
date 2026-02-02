import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from library.config import MARGIN_COLS, SHAPE_COLS, TEXTURE_COLS


class Float64Wrapper(BaseEstimator, TransformerMixin):
    """
    A transformer that casts the input data to float64 precision.
    This is crucial for minimizing numerical noise in the log loss metric calculation.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert to numpy array with float64 dtype
        return np.array(X, dtype=np.float64)


class FactorizedDiscriminantProjector(BaseEstimator, TransformerMixin):
    """
    Implements the Factorized Discriminative Bottleneck.

    This transformer splits the input features into three semantic groups:
    Margin, Shape, and Texture. For each group, it learns a specific
    transformation pipeline:
        PowerTransformer -> LinearDiscriminantAnalysis (Projection)

    The projected features from all groups are then concatenated.

    Parameters:
    -----------
    n_components : int
        The number of discriminative components to retain for each group.
        Default is 9.
    """

    def __init__(self, n_components=9):
        self.n_components = n_components

        # Initialize independent pipelines for each semantic group
        # We use make_pipeline to wrap PowerTransformer and LDA
        self.margin_pipe = make_pipeline(
            PowerTransformer(method="yeo-johnson"),
            LinearDiscriminantAnalysis(n_components=n_components),
        )
        self.shape_pipe = make_pipeline(
            PowerTransformer(method="yeo-johnson"),
            LinearDiscriminantAnalysis(n_components=n_components),
        )
        self.texture_pipe = make_pipeline(
            PowerTransformer(method="yeo-johnson"),
            LinearDiscriminantAnalysis(n_components=n_components),
        )

        # Store column definitions for easy access
        self.margin_cols = MARGIN_COLS
        self.shape_cols = SHAPE_COLS
        self.texture_cols = TEXTURE_COLS

        # Indices for numpy array fallback (assuming standard order: Margin, Shape, Texture)
        # Margin: 0-63, Shape: 64-127, Texture: 128-191
        self.margin_idx = slice(0, 64)
        self.shape_idx = slice(64, 128)
        self.texture_idx = slice(128, 192)

    def _get_group_data(self, X, group_name):
        """
        Helper to extract specific feature group data from X.
        Handles both DataFrame (by column name) and ndarray (by index).
        """
        if isinstance(X, pd.DataFrame):
            if group_name == "margin":
                # Ensure columns exist, otherwise fallback or raise
                return X[self.margin_cols].values
            elif group_name == "shape":
                return X[self.shape_cols].values
            elif group_name == "texture":
                return X[self.texture_cols].values
            else:
                raise ValueError(f"Unknown group name: {group_name}")
        else:
            # Assume numpy array with standard ordering
            # Note: This assumes the first 192 columns correspond to the provided features
            if group_name == "margin":
                return X[:, self.margin_idx]
            elif group_name == "shape":
                return X[:, self.shape_idx]
            elif group_name == "texture":
                return X[:, self.texture_idx]
            else:
                raise ValueError(f"Unknown group name: {group_name}")

    def fit(self, X, y):
        """
        Fits the internal pipelines for each feature group.

        Args:
            X: Input data (DataFrame or numpy array).
            y: Target labels (required for LDA).
        """
        # Extract data for each group
        X_margin = self._get_group_data(X, "margin")
        X_shape = self._get_group_data(X, "shape")
        X_texture = self._get_group_data(X, "texture")

        # Fit pipelines
        self.margin_pipe.fit(X_margin, y)
        self.shape_pipe.fit(X_shape, y)
        self.texture_pipe.fit(X_texture, y)

        return self

    def transform(self, X):
        """
        Transforms the input data by projecting each group and concatenating.

        Args:
            X: Input data.

        Returns:
            X_projected: Concatenated discriminative features (n_samples, n_components * 3).
        """
        # Extract data
        X_margin = self._get_group_data(X, "margin")
        X_shape = self._get_group_data(X, "shape")
        X_texture = self._get_group_data(X, "texture")

        # Transform
        proj_margin = self.margin_pipe.transform(X_margin)
        proj_shape = self.shape_pipe.transform(X_shape)
        proj_texture = self.texture_pipe.transform(X_texture)

        # Concatenate along feature axis
        X_projected = np.hstack([proj_margin, proj_shape, proj_texture])

        return X_projected.astype(np.float64)
