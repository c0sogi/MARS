import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    PolynomialFeatures,
)
from sklearn.pipeline import make_pipeline
from library.utils import ensure_float64


class MarginalBasis(BaseEstimator, TransformerMixin):
    """
    Applies Marginal Gaussianization using the Yeo-Johnson Power Transformer.
    Stabilizes variance feature-wise.
    """

    def __init__(self):
        self.transformer = PowerTransformer(method="yeo-johnson")

    def fit(self, X, y=None):
        X = ensure_float64(X)
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        X = ensure_float64(X)
        return self.transformer.transform(X)


class RotationalBasis(BaseEstimator, TransformerMixin):
    """
    Applies a Rotational Gaussianization pipeline:
    PowerTransformer -> PCA (no whitening) -> PowerTransformer.
    Aligns data with principal axes without the noise amplification of whitening,
    approximating Multivariate Normality.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pipeline = make_pipeline(
            PowerTransformer(method="yeo-johnson"),
            PCA(whiten=False, random_state=random_state),
            PowerTransformer(method="yeo-johnson"),
        )

    def fit(self, X, y=None):
        X = ensure_float64(X)
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        X = ensure_float64(X)
        return self.pipeline.transform(X)


class RobustBasis(BaseEstimator, TransformerMixin):
    """
    Applies Robust Gaussianization using a Quantile Transformer.
    Strictly handles skew and outliers using rank-based normalization.
    """

    def __init__(self, n_quantiles=50, random_state=42):
        self.n_quantiles = n_quantiles
        self.random_state = random_state
        self.transformer = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_quantiles,
            random_state=random_state,
        )

    def fit(self, X, y=None):
        X = ensure_float64(X)
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        X = ensure_float64(X)
        return self.transformer.transform(X)


class FactorizedInteractionProjector(BaseEstimator, TransformerMixin):
    """
    Implements the Factorized Interaction Scope.

    Mechanism:
    1. Splits the 192 Global features into Margin, Shape, and Texture groups.
    2. Projects each group onto a low-dimensional discriminative subspace using LDA.
    3. Synthesizes interactions between these subspaces using Polynomial Expansion.

    This explicitly models quadratic dependencies between biological domains
    (e.g., Serration x Texture) in a stable, dense subspace.
    """

    def __init__(self, n_components=10, interaction_only=False):
        self.n_components = n_components
        self.interaction_only = interaction_only

        # Internal discriminative models
        self.lda_margin = None
        self.lda_shape = None
        self.lda_texture = None
        self.poly = None

    def fit(self, X, y):
        """
        Fits the internal LDA models on feature subgroups and the Polynomial expansion.

        Args:
            X: Input array of shape (n_samples, 192).
            y: Target labels.
        """
        X = ensure_float64(X)

        # Slice features based on fixed dataset structure
        # Margin: 0-64, Shape: 64-128, Texture: 128-192
        X_m = X[:, 0:64]
        X_s = X[:, 64:128]
        X_t = X[:, 128:192]

        # Initialize and fit LDA projectors
        # We use a fixed number of components (k=10) to create the bottleneck
        self.lda_margin = LinearDiscriminantAnalysis(n_components=self.n_components)
        self.lda_shape = LinearDiscriminantAnalysis(n_components=self.n_components)
        self.lda_texture = LinearDiscriminantAnalysis(n_components=self.n_components)

        self.lda_margin.fit(X_m, y)
        self.lda_shape.fit(X_s, y)
        self.lda_texture.fit(X_t, y)

        # Transform to get bottleneck representation for Poly fitting
        z_m = self.lda_margin.transform(X_m)
        z_s = self.lda_shape.transform(X_s)
        z_t = self.lda_texture.transform(X_t)

        # Concatenate latent vectors
        Z = np.hstack([z_m, z_s, z_t])

        # Fit Polynomial Features to define the expansion map
        self.poly = PolynomialFeatures(
            degree=2, interaction_only=self.interaction_only, include_bias=False
        )
        self.poly.fit(Z)

        return self

    def transform(self, X):
        """
        Projects input data through the factorized topology.
        """
        X = ensure_float64(X)

        # Slice
        X_m = X[:, 0:64]
        X_s = X[:, 64:128]
        X_t = X[:, 128:192]

        # Project to bottleneck
        z_m = self.lda_margin.transform(X_m)
        z_s = self.lda_shape.transform(X_s)
        z_t = self.lda_texture.transform(X_t)

        # Concatenate
        Z = np.hstack([z_m, z_s, z_t])

        # Expand interactions
        return self.poly.transform(Z)
