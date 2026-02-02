import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer, PolynomialFeatures, normalize
from sklearn.decomposition import PCA
from library.config import Config
from library.utils import setup_logger
from library.data_loader import load_dataset
from library.embeddings import EmbeddingGenerator

# Initialize Logger
logger = setup_logger("feature_engineering")


class ContextAwareFusionTransformer(BaseEstimator, TransformerMixin):
    """
    Implements the Context-Aware Asymmetric Early Fusion (CAAEF) strategy.

    This transformer handles:
    1. Splitting the concatenated input into Anchor, Aux, and Metadata views.
    2. View 1 (Anchor): L2 Normalization.
    3. View 2 (Aux): PCA Compression -> L2 Normalization.
    4. View 3 (Metadata): RankGauss (Quantile) Scaling.
    5. View 4 (Interactions): Polynomial interaction between Top-K Aux PCA and Metadata -> RankGauss.
    6. Fusion: Concatenation of all processed views.
    """

    def __init__(
        self,
        anchor_dim=384,
        aux_dim=768,
        pca_components=Config.AUX_PCA_COMPONENTS,
        interaction_top_k=Config.INTERACTION_TOP_K,
        random_state=Config.SEED,
    ):
        self.anchor_dim = anchor_dim
        self.aux_dim = aux_dim
        self.pca_components = pca_components
        self.interaction_top_k = interaction_top_k
        self.random_state = random_state

        # Transformers
        self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
        self.meta_scaler = QuantileTransformer(
            output_distribution="normal", random_state=self.random_state
        )
        self.interaction_poly = PolynomialFeatures(
            degree=Config.INTERACTION_DEGREE, interaction_only=True, include_bias=False
        )
        self.interaction_scaler = QuantileTransformer(
            output_distribution="normal", random_state=self.random_state
        )

        # State to track metadata dimension determined at fit time
        self.meta_dim = None

    def _split_input(self, X):
        """
        Splits the concatenated input matrix into its constituent parts.
        Assumes structure: [Anchor (384) | Aux (768) | Metadata (N)]
        """
        # X is expected to be numpy array
        start_aux = self.anchor_dim
        start_meta = self.anchor_dim + self.aux_dim

        X_anchor = X[:, :start_aux]
        X_aux = X[:, start_aux:start_meta]
        X_meta = X[:, start_meta:]

        return X_anchor, X_aux, X_meta

    def fit(self, X, y=None):
        """
        Fits the internal transformers (PCA, Scalers) on the provided data.
        """
        X_anchor, X_aux, X_meta = self._split_input(X)
        self.meta_dim = X_meta.shape[1]

        # 1. Fit PCA on Aux
        self.pca.fit(X_aux)

        # 2. Fit Scaler on Metadata
        self.meta_scaler.fit(X_meta)

        # 3. Generate Interactions for fitting the Interaction Scaler
        # Transform Aux to get PCA components
        X_aux_pca = self.pca.transform(X_aux)

        # Select Top-K components
        # Ensure we don't select more than available
        k = min(self.interaction_top_k, self.pca_components)
        X_aux_top_k = X_aux_pca[:, :k]

        # Transform Metadata
        X_meta_scaled = self.meta_scaler.transform(X_meta)

        # Create interaction input: Concat Top-K PCA and Scaled Metadata
        # Note: We use scaled metadata for interactions to ensure stability
        X_inter_input = np.hstack([X_aux_top_k, X_meta_scaled])

        # Generate polynomials
        X_interactions = self.interaction_poly.fit_transform(X_inter_input)

        # Fit Scaler on Interactions
        self.interaction_scaler.fit(X_interactions)

        return self

    def transform(self, X):
        """
        Applies the transformations and fuses the views.
        """
        X_anchor, X_aux, X_meta = self._split_input(X)

        if self.meta_dim is not None and X_meta.shape[1] != self.meta_dim:
            # Warning could be logged here if dimensions mismatch, but we proceed
            pass

        # --- View 1: Anchor ---
        # L2 Normalize
        X_anchor_norm = normalize(X_anchor, norm="l2", axis=1)

        # --- View 2: Deep Semantics (Aux) ---
        # PCA -> L2 Normalize
        X_aux_pca = self.pca.transform(X_aux)
        X_aux_norm = normalize(X_aux_pca, norm="l2", axis=1)

        # --- View 3: Robust Metadata ---
        # RankGauss
        X_meta_scaled = self.meta_scaler.transform(X_meta)

        # --- View 4: Contextual Interactions ---
        # Inputs: Top-K PCA (from un-normalized PCA output) and Scaled Metadata
        k = min(self.interaction_top_k, self.pca_components)
        X_aux_top_k = X_aux_pca[:, :k]

        X_inter_input = np.hstack([X_aux_top_k, X_meta_scaled])
        X_interactions = self.interaction_poly.transform(X_inter_input)
        X_interactions_scaled = self.interaction_scaler.transform(X_interactions)

        # --- Fusion ---
        X_fused = np.hstack(
            [X_anchor_norm, X_aux_norm, X_meta_scaled, X_interactions_scaled]
        )

        return X_fused


def assemble_features(split, load_cached_data=True):
    """
    Loads necessary data components and assembles them into a single feature matrix
    suitable for the ContextAwareFusionTransformer.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached intermediate files.

    Returns:
        tuple: (X, y) where X is the concatenated feature matrix and y is the target (or None).
    """
    logger.info(f"Assembling features for split: {split}")

    # 1. Load Dataframe (Metadata + Numerics)
    df = load_dataset(split, load_cached_data=load_cached_data)

    # 2. Load Embeddings
    emb_gen = EmbeddingGenerator()
    X_anchor = emb_gen.get_embeddings(
        split, "anchor", load_cached_data=load_cached_data
    )
    X_aux = emb_gen.get_embeddings(split, "aux", load_cached_data=load_cached_data)

    # 3. Extract Metadata
    # Ensure columns are in correct order as defined in Config
    # We force float32 to save memory and ensure compatibility
    X_meta = df[Config.NUMERIC_COLS].values.astype(np.float32)

    # 4. Concatenate Features
    # Order: Anchor | Aux | Metadata
    # This order must match _split_input in the Transformer
    X = np.hstack([X_anchor, X_aux, X_meta])

    # 5. Extract Target
    y = None
    if Config.TARGET_COL in df.columns:
        y = df[Config.TARGET_COL].values.astype(int)

    logger.info(f"Assembled feature matrix shape: {X.shape}")

    return X, y
