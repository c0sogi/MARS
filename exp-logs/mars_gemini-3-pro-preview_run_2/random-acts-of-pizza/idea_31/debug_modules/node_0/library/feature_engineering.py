import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, normalize
from library.config import Config


class FusionTransformer(BaseEstimator, TransformerMixin):
    """
    Implements the Tri-Backbone Asymmetric Early Fusion (TBAEF) transformation logic.

    This transformer handles the specific preprocessing pipelines for four distinct
    feature views and fuses them into a single feature vector:

    1. Anchor View (MiniLM): L2 Normalization.
    2. Aux View 1 (MPNet): PCA Compression -> L2 Normalization.
    3. Aux View 2 (DistilRoBERTa): PCA Compression -> L2 Normalization.
    4. Metadata View: RankGauss Transformation (QuantileTransformer).
    """

    def __init__(self):
        """
        Initialize the internal transformers with reproducible seeds from Config.
        """
        self.pca_aux1 = PCA(
            n_components=Config.PCA_COMPONENTS, random_state=Config.SEED
        )
        self.pca_aux2 = PCA(
            n_components=Config.PCA_COMPONENTS, random_state=Config.SEED
        )
        self.meta_scaler = QuantileTransformer(
            output_distribution="normal", random_state=Config.SEED
        )

    def fit(self, X, y=None):
        """
        Fits the internal transformers (PCA and Scaler) on the provided feature dictionary.

        Args:
            X (dict): A dictionary containing the feature views:
                - 'anchor': np.ndarray of shape (n_samples, 384)
                - 'aux1': np.ndarray of shape (n_samples, 768)
                - 'aux2': np.ndarray of shape (n_samples, 768)
                - 'meta': np.ndarray of shape (n_samples, n_meta_features)
            y: Ignored (exists for scikit-learn compatibility).

        Returns:
            self: The fitted transformer instance.
        """
        # Fit PCA on the first auxiliary backbone (Deep Semantics)
        if "aux1" in X:
            self.pca_aux1.fit(X["aux1"])

        # Fit PCA on the second auxiliary backbone (Diverse Semantics)
        if "aux2" in X:
            self.pca_aux2.fit(X["aux2"])

        # Fit QuantileTransformer on the metadata
        if "meta" in X:
            self.meta_scaler.fit(X["meta"])

        return self

    def transform(self, X):
        """
        Applies the learned transformations and fuses the views.

        Args:
            X (dict): A dictionary containing the feature views (same structure as fit).

        Returns:
            np.ndarray: The concatenated feature matrix of shape (n_samples, total_features).
        """
        feature_list = []

        # 1. Anchor View: L2 Normalize (High-Res Foundation)
        if "anchor" in X:
            # normalize defaults to l2 norm along axis 1
            anchor_feat = normalize(X["anchor"], norm="l2")
            feature_list.append(anchor_feat)

        # 2. Aux View 1: PCA -> L2 Normalize (Deep Semantics Summary)
        if "aux1" in X:
            aux1_proj = self.pca_aux1.transform(X["aux1"])
            aux1_feat = normalize(aux1_proj, norm="l2")
            feature_list.append(aux1_feat)

        # 3. Aux View 2: PCA -> L2 Normalize (Diverse Semantics Summary)
        if "aux2" in X:
            aux2_proj = self.pca_aux2.transform(X["aux2"])
            aux2_feat = normalize(aux2_proj, norm="l2")
            feature_list.append(aux2_feat)

        # 4. Metadata View: RankGauss (Robust Numerical Features)
        if "meta" in X:
            meta_feat = self.meta_scaler.transform(X["meta"])
            feature_list.append(meta_feat)

        if not feature_list:
            raise ValueError("No valid feature views provided in input dictionary.")

        # Early Fusion: Concatenate all processed views horizontally
        fused_features = np.hstack(feature_list)

        return fused_features
