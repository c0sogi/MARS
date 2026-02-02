import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import setup_logger


class EmbeddingService:
    """
    Manages the computation and caching of text embeddings using Sentence Transformers.
    Supports 'anchor' (MiniLM) and 'aux' (MPNet) models.
    """

    def __init__(self):
        """
        Initialize the EmbeddingService.
        Sets up the logger and determines the computation device (CPU/GPU).
        """
        self.logger = setup_logger("embedding_service")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}  # Lazy loading cache for models

    def _get_cache_path(self, split_name: str, model_type: str) -> str:
        """
        Resolves the file path for the cached embeddings based on split and model type.
        """
        mapping = {
            "train": {
                "anchor": Config.TRAIN_EMB_ANCHOR,
                "aux": Config.TRAIN_EMB_AUX,
            },
            "val": {
                "anchor": Config.VAL_EMB_ANCHOR,
                "aux": Config.VAL_EMB_AUX,
            },
            "test": {
                "anchor": Config.TEST_EMB_ANCHOR,
                "aux": Config.TEST_EMB_AUX,
            },
        }

        if split_name not in mapping:
            raise ValueError(f"Unknown split_name: {split_name}")
        if model_type not in mapping[split_name]:
            raise ValueError(f"Unknown model_type: {model_type}")

        return mapping[split_name][model_type]

    def _get_model_name(self, model_type: str) -> str:
        """
        Resolves the Hugging Face model name based on the internal model type.
        """
        if model_type == "anchor":
            return Config.ANCHOR_MODEL_NAME
        elif model_type == "aux":
            return Config.AUX_MODEL_NAME
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def _load_model(self, model_type: str):
        """
        Loads the SentenceTransformer model into memory if not already loaded.
        """
        model_name = self._get_model_name(model_type)
        if model_name not in self.models:
            self.logger.info(f"Loading model: {model_name} on {self.device}...")
            model = SentenceTransformer(model_name, device=self.device)
            self.models[model_name] = model
        return self.models[model_name]

    def get_embeddings(
        self,
        df: pd.DataFrame,
        split_name: str,
        model_type: str,
        load_cached_data: bool = True,
    ) -> np.ndarray:
        """
        Retrieves embeddings for the provided dataframe.

        Logic:
        1. Check if cache exists and load_cached_data is True.
        2. If cached file exists, load it.
        3. Validate that cached embeddings match the number of rows in df (crucial for debug/subsampling).
        4. If cache miss or mismatch, compute embeddings from scratch.
        5. Save new embeddings to cache (unless in debug mismatch mode, we overwrite/save).

        Args:
            df (pd.DataFrame): Dataframe containing text columns.
            split_name (str): 'train', 'val', or 'test'.
            model_type (str): 'anchor' or 'aux'.
            load_cached_data (bool): Whether to attempt loading from disk.

        Returns:
            np.ndarray: Matrix of embeddings (n_samples, embedding_dim).
        """
        cache_path = self._get_cache_path(split_name, model_type)

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(cache_path):
            self.logger.info(
                f"Found cache for {split_name} ({model_type}) at {cache_path}. Loading..."
            )
            try:
                embeddings = np.load(cache_path)

                # Verify shape consistency (e.g., if we are in debug mode with fewer rows)
                if len(embeddings) == len(df):
                    self.logger.info("Cache loaded successfully and shape matches.")
                    return embeddings
                else:
                    self.logger.warning(
                        f"Cache shape mismatch! Cached: {len(embeddings)}, Current DF: {len(df)}. "
                        "Ignoring cache and recomputing."
                    )
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Recomputing...")
        else:
            if not load_cached_data:
                self.logger.info(
                    f"Cache loading disabled for {split_name} ({model_type})."
                )
            else:
                self.logger.info(f"Cache not found for {split_name} ({model_type}).")

        # 2. Compute Embeddings
        self.logger.info(f"Computing embeddings for {split_name} ({model_type})...")

        # Prepare text: Concatenate title and body
        # Note: data_loader.py ensures these columns exist and are strings (fillna handled)
        text_data = df[Config.TEXT_COLS].agg(" ".join, axis=1).tolist()

        model = self._load_model(model_type)

        # Encode
        embeddings = model.encode(
            text_data,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            device=self.device,
            normalize_embeddings=False,  # We handle normalization in the pipeline/feature engineering if needed
        )

        # 3. Save to Cache
        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embeddings)
        self.logger.info(f"Saved embeddings to {cache_path}. Shape: {embeddings.shape}")

        return embeddings
