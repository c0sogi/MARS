import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import QuantileTransformer, Normalizer
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from library.utils import setup_logger

logger = setup_logger("CustomTransformers")


class TensorSlicer(BaseEstimator, TransformerMixin):
    """
    Transformer to slice specific columns from a numpy array.
    Used to extract specific feature blocks (e.g., embeddings vs metadata)
    from a concatenated feature matrix.
    """

    def __init__(self, start_idx, end_idx):
        self.start_idx = start_idx
        self.end_idx = end_idx

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Check if indices are within bounds
        if X.shape[1] < self.end_idx:
            raise ValueError(
                f"Input has {X.shape[1]} features, but slice ends at {self.end_idx}."
            )
        return X[:, self.start_idx : self.end_idx]


class GMMTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that fits a Gaussian Mixture Model and transforms data
    into soft cluster membership probabilities (posterior probabilities).
    Acts as the 'Manifold Backbone'.
    """

    def __init__(self, n_components=10, covariance_type="diag", random_state=42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.gmm = None

    def fit(self, X, y=None):
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=100,
            n_init=1,
        )
        self.gmm.fit(X)
        return self

    def transform(self, X):
        if self.gmm is None:
            raise RuntimeError("GMMTransformer must be fitted before transform.")
        # Return the soft probabilities (density features)
        return self.gmm.predict_proba(X)


def build_feature_pipeline(
    anchor_dim=384, aux_dim=768, meta_dim=10, seed=42, pca_components=50
):
    """
    Constructs the MAADBE Feature Pipeline using FeatureUnion.

    Structure:
    1. View 1 (Anchor): Slice -> L2 Normalize
    2. View 2 (Deep Semantics): Slice -> PCA(n) -> L2 Normalize
    3. View 3 (Manifold): Slice -> PCA(n) -> GMM(10) -> RankGauss
    4. View 4 (Metadata): Slice -> RankGauss

    Args:
        anchor_dim (int): Dimension of the Anchor embedding (MiniLM).
        aux_dim (int): Dimension of the Auxiliary embedding (MPNet).
        meta_dim (int): Dimension of the metadata features.
        seed (int): Random seed for reproducibility.
        pca_components (int): Number of PCA components for semantic/manifold views.

    Returns:
        sklearn.pipeline.Pipeline: The constructed feature engineering pipeline.
    """

    # Calculate slice indices assuming input is [Anchor | Aux | Meta]
    idx_anchor_start = 0
    idx_anchor_end = anchor_dim

    idx_aux_start = idx_anchor_end
    idx_aux_end = idx_aux_start + aux_dim

    idx_meta_start = idx_aux_end
    idx_meta_end = idx_meta_start + meta_dim

    # View 1: Semantic Anchor (High-Res)
    # Just L2 normalize the raw embeddings
    anchor_pipeline = Pipeline(
        [
            ("slicer", TensorSlicer(idx_anchor_start, idx_anchor_end)),
            ("normalizer", Normalizer(norm="l2")),
        ]
    )

    # View 2: Deep Semantics (Compressed World Knowledge)
    # PCA compression + Normalization
    aux_semantic_pipeline = Pipeline(
        [
            ("slicer", TensorSlicer(idx_aux_start, idx_aux_end)),
            ("pca", PCA(n_components=pca_components, random_state=seed)),
            ("normalizer", Normalizer(norm="l2")),
        ]
    )

    # View 3: Manifold Density (Non-Linear Structure)
    # PCA -> GMM -> RankGauss
    # Note: We repeat PCA here to keep branches independent in FeatureUnion.
    # The computational cost is negligible compared to the benefits of modularity.
    manifold_pipeline = Pipeline(
        [
            ("slicer", TensorSlicer(idx_aux_start, idx_aux_end)),
            ("pca", PCA(n_components=pca_components, random_state=seed)),
            (
                "gmm",
                GMMTransformer(
                    n_components=10, covariance_type="diag", random_state=seed
                ),
            ),
            (
                "rank_gauss",
                QuantileTransformer(output_distribution="normal", random_state=seed),
            ),
        ]
    )

    # View 4: Robust Metadata
    # RankGauss scaling for numerical stability
    meta_pipeline = Pipeline(
        [
            ("slicer", TensorSlicer(idx_meta_start, idx_meta_end)),
            (
                "rank_gauss",
                QuantileTransformer(output_distribution="normal", random_state=seed),
            ),
        ]
    )

    # Combine all views
    feature_union = FeatureUnion(
        [
            ("view_anchor", anchor_pipeline),
            ("view_semantic", aux_semantic_pipeline),
            ("view_manifold", manifold_pipeline),
            ("view_metadata", meta_pipeline),
        ]
    )

    return feature_union
