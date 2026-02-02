import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, normalize
from library.config import Config


class OASFPreprocessor(BaseEstimator, TransformerMixin):
    """
    Implements the Orthogonalized Asymmetric Semantic Fusion (OASF) preprocessing strategy.

    This transformer expects the input X to be a horizontal concatenation of:
    [Anchor Embeddings (MiniLM) | Auxiliary Embeddings (MPNet) | Metadata Features]

    It performs the following operations:
    1. Splits the input into three views.
    2. L2 Normalizes the Anchor Embeddings.
    3. Orthogonalizes the Auxiliary Embeddings by regressing them on the Normalized Anchor Embeddings
       and computing the residuals. This isolates the unique semantic signal in the Auxiliary model.
    4. Compresses the residuals using PCA (Asymmetric Dimensionality Reduction).
    5. L2 Normalizes the compressed residuals.
    6. Applies QuantileTransformer (RankGauss) to the Metadata features.
    7. Fuses (concatenates) the three processed views.
    """

    def __init__(
        self,
        anchor_dim=Config.ANCHOR_DIM,
        aux_dim=Config.AUX_DIM,
        pca_components=Config.PCA_COMPONENTS,
        random_state=Config.RANDOM_SEED,
    ):
        """
        Args:
            anchor_dim (int): Dimensionality of the anchor backbone (default: 384).
            aux_dim (int): Dimensionality of the auxiliary backbone (default: 768).
            pca_components (int): Number of components for residual compression (default: 50).
            random_state (int): Seed for reproducibility.
        """
        self.anchor_dim = anchor_dim
        self.aux_dim = aux_dim
        self.pca_components = pca_components
        self.random_state = random_state

        # Internal Estimators
        # Ridge for Orthogonalization (Anchor -> Aux)
        # We use alpha=1.0 as a robust default for this feature orthogonalization task
        self.ridge = Ridge(alpha=1.0, random_state=self.random_state)

        # PCA for Asymmetric Compression of Residuals
        self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)

        # QuantileTransformer for Metadata (RankGauss)
        self.qt = QuantileTransformer(
            output_distribution="normal", random_state=self.random_state
        )

    def _split(self, X):
        """
        Splits the concatenated input matrix into Anchor, Aux, and Metadata views.
        """
        total_cols = X.shape[1]
        expected_min = self.anchor_dim + self.aux_dim

        if total_cols < expected_min:
            raise ValueError(
                f"Input X has {total_cols} columns, expected at least {expected_min} (Anchor+Aux)."
            )

        X_anchor = X[:, : self.anchor_dim]
        X_aux = X[:, self.anchor_dim : self.anchor_dim + self.aux_dim]
        X_meta = X[:, self.anchor_dim + self.aux_dim :]

        return X_anchor, X_aux, X_meta

    def fit(self, X, y=None):
        """
        Fits the internal transformers (Ridge, PCA, QuantileTransformer) on the provided data.
        """
        X_anchor, X_aux, X_meta = self._split(X)

        # 1. Normalize Anchor
        # We normalize anchor before using it for orthogonalization to match inference time behavior.
        # This ensures the regression learns the mapping from the actual features used in the final model.
        X_anchor_norm = normalize(X_anchor, norm="l2", axis=1)

        # 2. Orthogonalization: Fit Ridge to predict Aux from Normalized Anchor
        self.ridge.fit(X_anchor_norm, X_aux)

        # 3. Compute Residuals
        # The residuals represent the information in Aux that is NOT linearly explained by Anchor.
        X_aux_pred = self.ridge.predict(X_anchor_norm)
        residuals = X_aux - X_aux_pred

        # 4. Asymmetric Compression: Fit PCA on Residuals
        self.pca.fit(residuals)

        # 5. Metadata Processing: Fit QuantileTransformer
        if X_meta.shape[1] > 0:
            self.qt.fit(X_meta)

        return self

    def transform(self, X):
        """
        Applies the OASF transformation pipeline to the data.
        """
        X_anchor, X_aux, X_meta = self._split(X)

        # 1. View 1: Normalized Anchor
        X_anchor_norm = normalize(X_anchor, norm="l2", axis=1)

        # 2. View 2: Residual Semantics
        # Predict Aux from Normalized Anchor using the fitted Ridge model
        X_aux_pred = self.ridge.predict(X_anchor_norm)

        # Calculate Residuals
        residuals = X_aux - X_aux_pred

        # Compress Residuals using the fitted PCA
        residuals_compressed = self.pca.transform(residuals)

        # Normalize Compressed Residuals (Project onto hypersphere)
        residuals_norm = normalize(residuals_compressed, norm="l2", axis=1)

        # 3. View 3: Robust Metadata
        if X_meta.shape[1] > 0:
            X_meta_trans = self.qt.transform(X_meta)
        else:
            X_meta_trans = X_meta

        # Fusion: Concatenate all views
        X_fused = np.hstack([X_anchor_norm, residuals_norm, X_meta_trans])

        return X_fused
