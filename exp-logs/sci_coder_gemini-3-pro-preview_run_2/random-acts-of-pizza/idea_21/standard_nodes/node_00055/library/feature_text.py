import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from library.config import Config
from library.utils import setup_logger

# Initialize Logger
logger = setup_logger("feature_text")


class SBERTEmbedder:
    """
    Handles the generation of semantic text embeddings using Sentence-BERT.
    Implements L2 normalization and disk-based caching.
    """

    def __init__(self):
        """
        Initializes the SBERT model based on the configuration.
        """
        logger.info(f"Initializing SBERT model: {Config.SBERT_MODEL_NAME}")
        # Load the pre-trained SentenceTransformer model
        # We use the CPU or GPU automatically detected by the library
        self.model = SentenceTransformer(Config.SBERT_MODEL_NAME)

    def encode(self, texts: list) -> np.ndarray:
        """
        Encodes a list of text strings into embeddings and applies L2 normalization.

        Args:
            texts (list): List of strings to encode.

        Returns:
            np.ndarray: A numpy array of shape (n_samples, 384) containing normalized embeddings.
        """
        logger.info(f"Encoding {len(texts)} text samples...")

        # Generate embeddings
        # show_progress_bar=False to keep output clean as requested
        embeddings = self.model.encode(texts, show_progress_bar=False)

        # Apply L2 Normalization to project onto the hypersphere
        # This is crucial for cosine-similarity-like behavior in linear models
        logger.info("Applying L2 normalization to embeddings...")
        normalized_embeddings = normalize(embeddings, norm="l2", axis=1)

        return normalized_embeddings

    def process_and_cache(
        self, df: pd.DataFrame, cache_path: str, load_cached_data: bool = True
    ) -> np.ndarray:
        """
        Retrieves embeddings for the provided DataFrame.
        Checks the cache first; if missing or forced reload, computes and saves.

        Args:
            df (pd.DataFrame): DataFrame containing the 'combined_text' column.
            cache_path (str): Path to the .npy file for caching.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            np.ndarray: The embeddings array.
        """
        # Ensure the working directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            logger.info(f"Loading cached embeddings from {cache_path}")
            try:
                embeddings = np.load(cache_path)
                # Verify shape consistency with DataFrame
                if len(embeddings) == len(df):
                    return embeddings
                else:
                    logger.warning(
                        f"Cached embeddings shape {embeddings.shape} does not match "
                        f"DataFrame length {len(df)}. Recomputing..."
                    )
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from Scratch
        logger.info(f"Computing embeddings for {os.path.basename(cache_path)}...")

        # Ensure 'combined_text' exists
        if "combined_text" not in df.columns:
            raise KeyError("DataFrame missing required column: 'combined_text'")

        # Fill NaNs just in case, though data_loader should have handled it
        texts = df["combined_text"].fillna("").astype(str).tolist()

        embeddings = self.encode(texts)

        # 3. Save to Cache
        logger.info(f"Saving embeddings to {cache_path}")
        np.save(cache_path, embeddings)

        return embeddings
