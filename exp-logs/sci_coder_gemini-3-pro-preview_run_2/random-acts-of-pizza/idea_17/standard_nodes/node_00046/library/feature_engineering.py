import os
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, normalize
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import setup_logger


class Embedder:
    """
    Wraps the SBERT model to generate dense embeddings for text views.
    Handles caching of raw embeddings to disk to optimize runtime.
    """

    def __init__(self):
        self.logger = setup_logger("Embedder")
        self.model = None
        self.device = Config.DEVICE if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        """Lazy loading of the SBERT model to conserve memory until needed."""
        if self.model is None:
            self.logger.info(
                f"Loading SBERT model: {Config.SBERT_MODEL} on {self.device}"
            )
            self.model = SentenceTransformer(Config.SBERT_MODEL, device=self.device)

    def _get_cache_path(self, split: str, view_type: str) -> str:
        """Determines the cache path based on split and view type."""
        if view_type == "request":
            if split == "train":
                return Config.TRAIN_REQ_EMB_PATH
            if split == "val":
                return Config.VAL_REQ_EMB_PATH
            if split == "test":
                return Config.TEST_REQ_EMB_PATH
        elif view_type == "history":
            if split == "train":
                return Config.TRAIN_HIST_EMB_PATH
            if split == "val":
                return Config.VAL_HIST_EMB_PATH
            if split == "test":
                return Config.TEST_HIST_EMB_PATH
        else:
            raise ValueError(f"Unknown view_type: {view_type}")
        raise ValueError(f"Unknown split: {split}")

    def get_embeddings(
        self,
        df: pd.DataFrame,
        split: str,
        view_type: str,
        load_cached_data: bool = True,
    ) -> np.ndarray:
        """
        Generates or loads embeddings for a specific view and split.

        Args:
            df (pd.DataFrame): DataFrame containing the text columns ('text_view' or 'history_view').
            split (str): 'train', 'val', or 'test'.
            view_type (str): 'request' or 'history'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            np.ndarray: Matrix of embeddings (N, 384).
        """
        cache_path = self._get_cache_path(split, view_type)

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            self.logger.info(
                f"Loading cached {view_type} embeddings for {split} from {cache_path}"
            )
            try:
                return np.load(cache_path)
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        self.logger.info(f"Computing {view_type} embeddings for {split}...")
        self._load_model()

        if view_type == "request":
            texts = df["text_view"].fillna("").astype(str).tolist()
        elif view_type == "history":
            texts = df["history_view"].fillna("").astype(str).tolist()
        else:
            raise ValueError(f"Invalid view_type: {view_type}")

        # Encode
        # batch_size=32 is generally safe for 384d models on standard GPUs
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # We normalize later in ViewTransformer
        )

        # 3. Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embeddings)
        self.logger.info(f"Saved {view_type} embeddings to {cache_path}")

        return embeddings


class ViewTransformer:
    """
    Handles the asymmetric transformation and fusion of features.
    Fits PCA and QuantileTransformer on training data, applies to all.
    """

    def __init__(self):
        self.logger = setup_logger("ViewTransformer")
        self.pca = PCA(
            n_components=Config.HISTORY_PCA_COMPONENTS, random_state=Config.SEED
        )
        self.qt = QuantileTransformer(
            output_distribution="normal", random_state=Config.SEED
        )
        self.is_fitted = False

    def fit(self, X_hist: np.ndarray, X_meta: np.ndarray):
        """
        Fits the internal transformers (PCA for history, QT for metadata).
        This should be called on the training fold data only.

        Args:
            X_hist (np.ndarray): Raw history embeddings (N, 384).
            X_meta (np.ndarray): Raw numerical metadata (N, F).
        """
        self.logger.info("Fitting ViewTransformer...")

        # Fit PCA on History View (Asymmetric Compression)
        self.pca.fit(X_hist)

        # Fit QuantileTransformer on Metadata (Robust Scaling)
        self.qt.fit(X_meta)

        self.is_fitted = True
        return self

    def transform(
        self, X_req: np.ndarray, X_hist: np.ndarray, X_meta: np.ndarray
    ) -> np.ndarray:
        """
        Applies transformations and fuses views.

        Args:
            X_req (np.ndarray): Raw request embeddings (N, 384).
            X_hist (np.ndarray): Raw history embeddings (N, 384).
            X_meta (np.ndarray): Raw numerical metadata (N, F).

        Returns:
            np.ndarray: Fused feature matrix.
        """
        if not self.is_fitted:
            raise RuntimeError("ViewTransformer must be fitted before transform.")

        # View 1: Request -> L2 Normalize (Project to hypersphere)
        # SBERT embeddings are directional; L2 norm makes dot product equivalent to cosine similarity.
        X_req_norm = normalize(X_req, norm="l2", axis=1)

        # View 2: History -> PCA -> L2 Normalize
        # Compress history to low-dim persona vector, then normalize.
        X_hist_pca = self.pca.transform(X_hist)
        X_hist_norm = normalize(X_hist_pca, norm="l2", axis=1)

        # View 3: Metadata -> RankGauss (QuantileTransformer)
        # Enforce Gaussian distribution for linear model compatibility.
        X_meta_trans = self.qt.transform(X_meta)

        # Fusion: Concatenate all views
        X_fused = np.hstack([X_req_norm, X_hist_norm, X_meta_trans])

        return X_fused
