import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, normalize
from library.config import Config
from library.utils import setup_logger


class CoherenceFeatureProcessor:
    """
    Implements the Coherence-Augmented Multi-Field feature processing logic.
    Handles dimensionality reduction, scaling, and feature fusion while ensuring
    statistical isolation of the validation set (transformations are fitted on train only).
    """

    def __init__(self):
        self.logger = setup_logger(
            "FeatureProcessor", "./working/idea_40/feature_engine.log"
        )

        # View 3: Global Context Compression
        self.pca = PCA(n_components=Config.PCA_COMPONENTS, random_state=Config.SEED)

        # View 4: Coherence Score Normalization (RankGauss)
        self.coherence_scaler = QuantileTransformer(
            output_distribution="normal", random_state=Config.SEED
        )

        # View 5: Metadata Normalization (RankGauss)
        self.meta_scaler = QuantileTransformer(
            output_distribution="normal", random_state=Config.SEED
        )

    def _compute_coherence(
        self, title_emb: np.ndarray, body_emb: np.ndarray
    ) -> np.ndarray:
        """
        Computes the cosine similarity between L2-normalized title and body embeddings.
        Input embeddings must already be L2 normalized.
        Returns array of shape (n_samples, 1).
        """
        # Dot product of normalized vectors equals cosine similarity
        # Element-wise multiplication followed by sum across dimensions
        dot_product = np.sum(title_emb * body_emb, axis=1, keepdims=True)
        return dot_product

    def fit(
        self,
        title_emb: np.ndarray,
        body_emb: np.ndarray,
        global_emb: np.ndarray,
        meta_features: np.ndarray,
    ):
        """
        Fits the internal transformers (PCA, Scalers) on the training data.

        Args:
            title_emb: Raw MiniLM embeddings for request_title (N, 384)
            body_emb: Raw MiniLM embeddings for request_text (N, 384)
            global_emb: Raw MPNet embeddings for concatenated text (N, 768)
            meta_features: Raw numerical metadata (N, D)
        """
        self.logger.info("Fitting CoherenceFeatureProcessor...")

        # 1. Fit PCA on Global Context (View 3)
        self.pca.fit(global_emb)

        # 2. Fit Scaler on Coherence Score (View 4)
        # We must normalize first to compute valid cosine similarity
        norm_title = normalize(title_emb, norm="l2")
        norm_body = normalize(body_emb, norm="l2")
        coherence_scores = self._compute_coherence(norm_title, norm_body)
        self.coherence_scaler.fit(coherence_scores)

        # 3. Fit Scaler on Metadata (View 5)
        self.meta_scaler.fit(meta_features)

        return self

    def transform(
        self,
        title_emb: np.ndarray,
        body_emb: np.ndarray,
        global_emb: np.ndarray,
        meta_features: np.ndarray,
    ) -> np.ndarray:
        """
        Applies transformations and fuses all views into a single feature matrix.

        Args:
            title_emb: Raw MiniLM embeddings (N, 384)
            body_emb: Raw MiniLM embeddings (N, 384)
            global_emb: Raw MPNet embeddings (N, 768)
            meta_features: Raw numerical metadata (N, D)

        Returns:
            X_fused: Concatenated feature matrix (N, Total_Dims)
        """
        # View 1: Title Semantics (L2 Normalized)
        norm_title = normalize(title_emb, norm="l2")

        # View 2: Body Semantics (L2 Normalized)
        norm_body = normalize(body_emb, norm="l2")

        # View 3: Global Context (PCA -> L2 Normalized)
        # Apply PCA projection
        global_pca = self.pca.transform(global_emb)
        # L2 normalize after projection to map to hypersphere
        norm_global = normalize(global_pca, norm="l2")

        # View 4: Semantic Coherence (Cosine Sim -> RankGauss)
        coherence_scores = self._compute_coherence(norm_title, norm_body)
        trans_coherence = self.coherence_scaler.transform(coherence_scores)

        # View 5: Robust Metadata (RankGauss)
        trans_meta = self.meta_scaler.transform(meta_features)

        # Fusion: Concatenate all views
        X_fused = np.hstack(
            [
                norm_title,  # 384 dims
                norm_body,  # 384 dims
                norm_global,  # 50 dims
                trans_coherence,  # 1 dim
                trans_meta,  # ~10 dims
            ]
        )

        return X_fused
