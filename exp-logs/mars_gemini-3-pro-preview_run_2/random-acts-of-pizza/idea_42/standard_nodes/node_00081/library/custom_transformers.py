import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from library import config


class ArraySelector(BaseEstimator, TransformerMixin):
    """
    Transformer that selects a slice of columns from a numpy array.
    Useful when working with concatenated feature sets in a Pipeline.
    """

    def __init__(self, start_index, end_index=None):
        """
        Args:
            start_index (int): The starting column index (inclusive).
            end_index (int, optional): The ending column index (exclusive).
                                       If None, selects up to the last column.
        """
        self.start_index = start_index
        self.end_index = end_index

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure X is a numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.end_index is not None:
            return X[:, self.start_index : self.end_index]
        else:
            return X[:, self.start_index :]


class WhitenedPCANormalizer(BaseEstimator, TransformerMixin):
    """
    Applies PCA with whitening enabled, followed immediately by L2 normalization.

    This specific sequence ensures that:
    1. Whitening: Normalizes the variance of the principal components, scaling up
       subtle semantic signals found in lower eigenvalues.
    2. L2 Normalization: Projects the resulting whitened vector onto the unit hypersphere.
       This prevents the auxiliary view from having an arbitrary magnitude scale compared
       to other normalized embedding views (like MiniLM anchors), while preserving the
       relative variance adjustments made by the whitening step.
    """

    def __init__(self, n_components=config.PCA_COMPONENTS):
        """
        Args:
            n_components (int): Number of principal components to keep.
                                Defaults to value in config.py.
        """
        self.n_components = n_components
        self.pca = PCA(
            n_components=self.n_components, whiten=True, random_state=config.SEED
        )

    def fit(self, X, y=None):
        """
        Fits the internal PCA model on X.
        """
        self.pca.fit(X)
        return self

    def transform(self, X):
        """
        Applies Whitened PCA followed by L2 Normalization.
        """
        # 1. Apply PCA (Whitened)
        X_pca = self.pca.transform(X)

        # 2. Apply L2 Normalization
        # We use the functional API for efficiency
        X_norm = normalize(X_pca, norm="l2", axis=1)

        return X_norm
