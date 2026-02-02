import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import setup_logger


class FeatureGenerator:
    """
    Handles the generation of features from raw data.
    Specific responsibilities:
    1. Generate L2-normalized text embeddings using a pre-trained Transformer.
    2. Extract raw tabular metadata features for downstream processing.
    3. Manage caching of computationally expensive embeddings.
    """

    def __init__(self):
        self.logger = setup_logger("FeatureGenerator")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = Config.MODEL_NAME

        # Initialize model placeholder
        self._model = None

    def _get_model(self):
        """Lazy loading of the Sentence Transformer model."""
        if self._model is None:
            self.logger.info(
                f"Loading SentenceTransformer model: {self.model_name} on {self.device}"
            )
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def generate_embeddings(
        self, df: pd.DataFrame, split_name: str, load_cached_data: bool = True
    ) -> np.ndarray:
        """
        Generates or loads text embeddings for a given dataframe.

        Args:
            df (pd.DataFrame): Data containing text columns.
            split_name (str): Name of the split (e.g., 'train', 'val', 'test') for caching.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            np.ndarray: L2-normalized embeddings of shape (n_samples, embedding_dim).
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        cache_path = os.path.join(Config.WORKING_DIR, f"{split_name}_embeddings.npy")

        # 1. Try Loading from Cache
        if load_cached_data:
            if os.path.exists(cache_path):
                self.logger.info(
                    f"Loading {split_name} embeddings from cache: {cache_path}"
                )
                try:
                    embeddings = np.load(cache_path)
                    if embeddings.shape[0] == len(df):
                        return embeddings
                    else:
                        self.logger.warning(
                            f"Cached embeddings shape {embeddings.shape} does not match dataframe length {len(df)}. "
                            "Recomputing..."
                        )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load cache {cache_path}: {e}. Recomputing..."
                    )
            else:
                self.logger.info(
                    f"Cache file not found for {split_name}. Computing from scratch..."
                )

        # 2. Compute Embeddings
        self.logger.info(
            f"Computing embeddings for {split_name} ({len(df)} samples)..."
        )

        # Concatenate text columns as defined in Config
        # We assume data cleaning (filling NaNs) has been done by DataLoader
        text_data = df[Config.TEXT_COLS[0]].astype(str)
        for col in Config.TEXT_COLS[1:]:
            text_data = text_data + " " + df[col].astype(str)

        sentences = text_data.tolist()

        model = self._get_model()

        # Encode sentences
        # show_progress_bar=False to keep output clean as per requirements
        raw_embeddings = model.encode(
            sentences,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # We normalize manually to be explicit
        )

        # 3. L2 Normalization
        # Project embeddings onto the hypersphere
        self.logger.info("Applying L2 normalization to embeddings...")
        normalized_embeddings = normalize(raw_embeddings, norm="l2", axis=1)

        # 4. Save to Cache
        self.logger.info(f"Saving embeddings to cache: {cache_path}")
        try:
            np.save(cache_path, normalized_embeddings)
        except Exception as e:
            self.logger.warning(f"Failed to save embeddings to cache: {e}")

        return normalized_embeddings

    def extract_tabular_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extracts the numeric metadata features defined in Config.

        Args:
            df (pd.DataFrame): Data containing numeric columns.

        Returns:
            np.ndarray: Array of numeric features of shape (n_samples, n_numeric_features).
        """
        self.logger.info("Extracting tabular metadata features...")

        # Validate columns exist
        missing_cols = [col for col in Config.NUMERIC_COLS if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"The following required numeric columns are missing in the dataframe: {missing_cols}"
            )

        # Extract and convert to numpy float32
        features = df[Config.NUMERIC_COLS].to_numpy(dtype=np.float32)

        return features
