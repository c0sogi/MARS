import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from library.utils import setup_logger, get_device, set_seed
from library.data_loader import NUMERICAL_FEATURES, WORKING_DIR


class EmbeddingGenerator:
    """
    Handles the generation of semantic embeddings using multiple SentenceTransformer models.
    Implements caching to avoid redundant computation.
    """

    def __init__(self):
        self.device = get_device()
        self.logger = setup_logger(
            "EmbeddingGenerator", os.path.join(WORKING_DIR, "feature_extractor.log")
        )

        # Define model names
        self.high_res_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.low_res_model_name = "sentence-transformers/all-mpnet-base-v2"

        # Models are loaded lazily to save memory
        self.high_res_model = None
        self.low_res_model = None

    def _load_model(self, model_name):
        """
        Loads a SentenceTransformer model onto the configured device.
        """
        self.logger.info(f"Loading model: {model_name} on {self.device}")
        return SentenceTransformer(model_name, device=str(self.device))

    def process_split(self, df, split_name, load_cached_data=True, batch_size=64):
        """
        Generates or loads embeddings for a specific data split.

        Args:
            df (pd.DataFrame): The dataframe containing the text data.
            split_name (str): Name of the split (e.g., 'train', 'val', 'test').
            load_cached_data (bool): Whether to attempt loading from cache.
            batch_size (int): Batch size for inference.

        Returns:
            tuple: (embeddings_high_res, embeddings_low_res) as numpy arrays.
        """
        # Ensure working directory exists
        os.makedirs(WORKING_DIR, exist_ok=True)

        # Define cache paths
        path_high = os.path.join(WORKING_DIR, f"{split_name}_embeddings_high_res.npy")
        path_low = os.path.join(WORKING_DIR, f"{split_name}_embeddings_low_res.npy")

        # Check if cache exists
        cache_exists = os.path.exists(path_high) and os.path.exists(path_low)

        if load_cached_data and cache_exists:
            self.logger.info(
                f"Loading cached embeddings for '{split_name}' from {WORKING_DIR}..."
            )
            emb_high = np.load(path_high)
            emb_low = np.load(path_low)
        else:
            self.logger.info(
                f"Generating embeddings for '{split_name}' (Cache Miss or Force Reload)..."
            )

            # Ensure text column is string and handle missing values
            texts = df["text_combined"].fillna("").astype(str).tolist()

            # --- High Resolution Model (MiniLM) ---
            self.logger.info(f"Encoding High-Res Features ({len(texts)} samples)...")
            if self.high_res_model is None:
                self.high_res_model = self._load_model(self.high_res_model_name)

            emb_high = self.high_res_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,  # Normalization happens in training pipeline
            )
            np.save(path_high, emb_high)

            # Free memory
            del self.high_res_model
            self.high_res_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # --- Low Resolution Model (MPNet) ---
            self.logger.info(f"Encoding Low-Res Features ({len(texts)} samples)...")
            if self.low_res_model is None:
                self.low_res_model = self._load_model(self.low_res_model_name)

            emb_low = self.low_res_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            np.save(path_low, emb_low)

            # Free memory
            del self.low_res_model
            self.low_res_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info(f"Embeddings saved to {WORKING_DIR}")

        return emb_high, emb_low


def extract_metadata_features(df):
    """
    Extracts numerical metadata features from the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        np.ndarray: Array of numerical features (n_samples, n_features).
    """
    # Ensure all required columns are present (data_loader should handle this, but safety check)
    missing_cols = [col for col in NUMERICAL_FEATURES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")

    # Select features
    features = df[NUMERICAL_FEATURES].copy()

    # Fill missing values with 0.0
    features = features.fillna(0.0)

    return features.values.astype(np.float32)
