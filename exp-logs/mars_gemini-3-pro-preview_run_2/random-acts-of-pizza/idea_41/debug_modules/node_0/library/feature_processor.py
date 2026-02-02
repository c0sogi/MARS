import os
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, normalize
from library.config import Config
from library.utils import setup_logger


class WhitenedFusionPipeline:
    """
    Implements the Whitened Multi-Field Asymmetric Dual-Backbone feature fusion strategy.

    This pipeline:
    1. Splits Anchor embeddings into Title and Body views (L2 Normalized).
    2. Compresses Auxiliary embeddings using Whitened PCA (then L2 Normalized).
    3. Transforms Metadata using RankGauss (QuantileTransformer).
    4. Concatenates all views into a single feature vector.
    """

    def __init__(self):
        self.logger = setup_logger("WhitenedFusionPipeline")

        # Hyperparameters from Config
        self.n_components = Config.PCA_N_COMPONENTS
        self.whiten = Config.PCA_WHITEN
        self.scaler_dist = Config.SCALER_OUTPUT_DISTRIBUTION
        self.seed = Config.SEED

        # State containers
        self.pca = None
        self.scaler = None
        self.metadata_cols = [
            "unix_timestamp_of_request",
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_comments_in_raop_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_posts_on_raop_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",
        ]

    def fit(
        self, anchor_emb: np.ndarray, aux_emb: np.ndarray, metadata_df: pd.DataFrame
    ):
        """
        Fits the internal transformers (PCA and Scaler) on the training data.

        Args:
            anchor_emb (np.ndarray): Anchor embeddings [N, 768] (Title+Body).
            aux_emb (np.ndarray): Auxiliary embeddings [N, 768] (Global).
            metadata_df (pd.DataFrame): DataFrame containing numerical metadata.
        """
        self.logger.info("Fitting WhitenedFusionPipeline...")

        # 1. Fit PCA on Auxiliary View (Global Context)
        # We use Whitening to equalize variance across components
        self.logger.info(
            f"Fitting PCA (n={self.n_components}, whiten={self.whiten}) on Aux embeddings..."
        )
        self.pca = PCA(
            n_components=self.n_components, whiten=self.whiten, random_state=self.seed
        )
        self.pca.fit(aux_emb)

        # 2. Fit Scaler on Metadata
        self.logger.info(
            f"Fitting QuantileTransformer (dist={self.scaler_dist}) on Metadata..."
        )
        meta_matrix = self._extract_metadata_matrix(metadata_df)
        self.scaler = QuantileTransformer(
            output_distribution=self.scaler_dist, random_state=self.seed
        )
        self.scaler.fit(meta_matrix)

        self.logger.info("Pipeline fitting complete.")
        return self

    def transform(
        self, anchor_emb: np.ndarray, aux_emb: np.ndarray, metadata_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Transforms input data into the fused feature space.

        Args:
            anchor_emb (np.ndarray): Anchor embeddings [N, 768].
            aux_emb (np.ndarray): Auxiliary embeddings [N, 768].
            metadata_df (pd.DataFrame): Metadata DataFrame.

        Returns:
            np.ndarray: Fused feature matrix.
        """
        if self.pca is None or self.scaler is None:
            raise RuntimeError("Pipeline must be fitted before calling transform.")

        # --- View 1 & 2: Anchor (Title & Body) ---
        # Anchor embeddings are [Title (384) | Body (384)]
        # We split them to normalize independently
        half_dim = anchor_emb.shape[1] // 2
        title_emb = anchor_emb[:, :half_dim]
        body_emb = anchor_emb[:, half_dim:]

        # L2 Normalize Title
        title_norm = normalize(title_emb, norm="l2", axis=1)

        # L2 Normalize Body
        body_norm = normalize(body_emb, norm="l2", axis=1)

        # --- View 3: Auxiliary (Global Context) ---
        # Project using Whitened PCA
        aux_pca = self.pca.transform(aux_emb)

        # L2 Normalize *after* Whitening (as per Idea)
        # This ensures the auxiliary view has unit norm like the anchors
        aux_norm = normalize(aux_pca, norm="l2", axis=1)

        # --- View 4: Metadata ---
        meta_matrix = self._extract_metadata_matrix(metadata_df)
        meta_trans = self.scaler.transform(meta_matrix)

        # --- Fusion ---
        # Concatenate all views
        fused_features = np.hstack([title_norm, body_norm, aux_norm, meta_trans])

        return fused_features

    def _extract_metadata_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Helper to extract specific numerical columns from DataFrame.
        """
        # Ensure all columns exist, fill missing with 0 if necessary (though DataLoader handles this)
        missing = [c for c in self.metadata_cols if c not in df.columns]
        if missing:
            self.logger.warning(f"Missing metadata columns: {missing}. Filling with 0.")
            for c in missing:
                df[c] = 0.0

        return df[self.metadata_cols].values.astype(np.float32)

    def save(self, filepath: str):
        """
        Saves the fitted pipeline state to disk.
        """
        state = {
            "pca": self.pca,
            "scaler": self.scaler,
            "metadata_cols": self.metadata_cols,
        }
        joblib.dump(state, filepath)
        self.logger.info(f"Pipeline state saved to {filepath}")

    def load(self, filepath: str):
        """
        Loads the pipeline state from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")

        state = joblib.load(filepath)
        self.pca = state["pca"]
        self.scaler = state["scaler"]
        self.metadata_cols = state.get("metadata_cols", self.metadata_cols)
        self.logger.info(f"Pipeline state loaded from {filepath}")
        return self
