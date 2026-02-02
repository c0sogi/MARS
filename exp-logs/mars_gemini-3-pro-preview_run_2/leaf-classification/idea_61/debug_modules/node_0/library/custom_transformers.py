import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
from library.config import FLOAT_PRECISION


class FactorizedDiscriminantProjector(BaseEstimator, TransformerMixin):
    """
    Implements the Factorized-Discriminative Manifold projection for the FDME strategy.

    This transformer:
    1. Splits the 192-dim Global View into Margin, Shape, and Texture semantic groups.
    2. Projects each group into a low-dimensional discriminative subspace using
       Regularized LDA (Factorization).
    3. Computes pairwise cross-domain interactions (Margin-Texture, Shape-Texture,
       Margin-Shape) using Polynomial expansion to capture biological dependencies.
    """

    def __init__(self, n_components=9, solver="lsqr", shrinkage="auto"):
        """
        Args:
            n_components (int): Number of discriminative components to project each group onto.
            solver (str): LDA solver ('lsqr' or 'eigen' for shrinkage support).
            shrinkage (float or str): Regularization parameter for LDA.
        """
        self.n_components = n_components
        self.solver = solver
        self.shrinkage = shrinkage

        # Internal state for fitted projectors
        self.lda_margin_ = None
        self.lda_shape_ = None
        self.lda_texture_ = None

        # Interaction synthesizer
        # interaction_only=True: generates x_i, x_j, x_i*x_j. Excludes x_i^2.
        # include_bias=False: LDA centers data, so bias term is redundant.
        self.poly = PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=False
        )

    def fit(self, X, y):
        """
        Fits the independent LDA projectors on the semantic feature groups.

        Args:
            X: Input data of shape (n_samples, 192).
            y: Target labels.
        """
        # Validate inputs and ensure double precision
        X, y = check_X_y(X, y, dtype=FLOAT_PRECISION)

        # Feature Slicing based on fixed dataset schema
        # Margin: 0-64, Shape: 64-128, Texture: 128-192
        X_margin = X[:, 0:64]
        X_shape = X[:, 64:128]
        X_texture = X[:, 128:192]

        # Initialize LDA models with specified hyperparameters
        self.lda_margin_ = LinearDiscriminantAnalysis(
            solver=self.solver, shrinkage=self.shrinkage, n_components=self.n_components
        )
        self.lda_shape_ = LinearDiscriminantAnalysis(
            solver=self.solver, shrinkage=self.shrinkage, n_components=self.n_components
        )
        self.lda_texture_ = LinearDiscriminantAnalysis(
            solver=self.solver, shrinkage=self.shrinkage, n_components=self.n_components
        )

        # Fit each projector on its specific domain
        self.lda_margin_.fit(X_margin, y)
        self.lda_shape_.fit(X_shape, y)
        self.lda_texture_.fit(X_texture, y)

        return self

    def transform(self, X):
        """
        Projects semantic groups and synthesizes cross-domain interactions.

        Args:
            X: Input data of shape (n_samples, 192).

        Returns:
            X_out: Transformed feature matrix with interaction terms (float64).
        """
        # Check is fitted
        check_is_fitted(self, ["lda_margin_", "lda_shape_", "lda_texture_"])

        # Validate input
        X = check_array(X, dtype=FLOAT_PRECISION)

        # 1. Slice
        X_margin = X[:, 0:64]
        X_shape = X[:, 64:128]
        X_texture = X[:, 128:192]

        # 2. Project to Discriminative Subspace
        # Each projection results in (n_samples, n_components)
        P_margin = self.lda_margin_.transform(X_margin).astype(FLOAT_PRECISION)
        P_shape = self.lda_shape_.transform(X_shape).astype(FLOAT_PRECISION)
        P_texture = self.lda_texture_.transform(X_texture).astype(FLOAT_PRECISION)

        # 3. Interaction Synthesis
        # We explicitly pair domains to capture specific biological couplings

        # Pair A: Margin (x) Texture
        # Concatenate projections then expand
        pair_mt = np.hstack([P_margin, P_texture])
        inter_mt = self.poly.fit_transform(pair_mt)

        # Pair B: Shape (x) Texture
        pair_st = np.hstack([P_shape, P_texture])
        inter_st = self.poly.fit_transform(pair_st)

        # Pair C: Margin (x) Shape
        pair_ms = np.hstack([P_margin, P_shape])
        inter_ms = self.poly.fit_transform(pair_ms)

        # 4. Final Concatenation
        # Stack all interaction views horizontally
        X_out = np.hstack([inter_mt, inter_st, inter_ms])

        return X_out.astype(FLOAT_PRECISION)
