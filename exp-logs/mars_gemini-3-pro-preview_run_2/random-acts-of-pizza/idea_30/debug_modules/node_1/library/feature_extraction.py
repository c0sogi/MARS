import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import setup_logger, save_object, load_object
from library.data_loader import load_and_process_data

# Initialize logger
logger = setup_logger("feature_extraction")


class EmbeddingManager:
    """
    Manages the generation and caching of text embeddings using SentenceTransformers.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"EmbeddingManager initialized on device: {self.device}")

    def get_embeddings(self, texts, model_name, cache_path, load_cached=True):
        """
        Generates or loads embeddings for a list of texts.

        Args:
            texts (list or np.array): List of text strings to encode.
            model_name (str): HuggingFace model identifier.
            cache_path (str): Path to save/load the numpy array.
            load_cached (bool): Whether to attempt loading from disk.

        Returns:
            np.ndarray: Matrix of embeddings (N_samples, Embedding_dim).
        """
        # 1. Try Loading from Cache
        if load_cached and os.path.exists(cache_path):
            logger.info(f"Loading embeddings from cache: {cache_path}")
            try:
                embeddings = np.load(cache_path)
                # Verify shape matches text count
                if embeddings.shape[0] == len(texts):
                    logger.info(
                        f"Successfully loaded embeddings. Shape: {embeddings.shape}"
                    )
                    return embeddings
                else:
                    logger.warning(
                        f"Cached embeddings shape {embeddings.shape} does not match text count {len(texts)}. Recomputing..."
                    )
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}. Recomputing...")

        # 2. Compute Embeddings
        logger.info(f"Computing embeddings with model: {model_name}...")

        # Load model
        try:
            model = SentenceTransformer(model_name, device=self.device)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

        # Encode
        # batch_size=32 is a safe default for most GPUs with these model sizes
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # We handle normalization in the pipeline/model
        )

        logger.info(f"Embeddings computed. Shape: {embeddings.shape}")

        # 3. Save to Cache
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, embeddings)
            logger.info(f"Saved embeddings to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save embeddings to {cache_path}: {e}")

        return embeddings


class FeaturePreprocessor:
    """
    Orchestrates the loading of data, generation of embeddings, and assembly of the
    multi-view feature matrix.
    """

    def __init__(self):
        self.embedder = EmbeddingManager()

    def _get_cache_path(self, base_path, debug):
        """Helper to modify cache path for debug mode."""
        if not debug:
            return base_path

        base, ext = os.path.splitext(base_path)
        return f"{base}_debug{ext}"

    def get_data(self, split="train", load_cached=True, debug=False):
        """
        Retrieves the full feature set for a specific data split.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached (bool): Whether to use cached data/embeddings.
            debug (bool): If True, uses a small subset of data.

        Returns:
            dict: Dictionary containing:
                - 'X': Combined feature matrix (np.ndarray).
                - 'y': Labels (np.ndarray) or None for test.
                - 'ids': Request IDs (np.ndarray).
                - 'feature_slices': Dict mapping view names to slice objects.
        """
        logger.info(f"Preparing data for split: {split} (Debug={debug})")

        # 1. Load Processed Dataframe (Text + Metadata)
        df = load_and_process_data(
            split=split, load_cached_data=load_cached, debug=debug
        )

        # Extract Texts
        texts = df["text_combined"].fillna("").tolist()
        ids = df["request_id"].values

        # Extract Labels if available
        y = None
        if "label" in df.columns:
            y = df["label"].values

        # 2. Generate/Load Embeddings

        # Determine Cache Paths based on split and debug status
        if split == "train":
            path_primary = Config.TRAIN_EMBEDDINGS_PRIMARY
            path_aux = Config.TRAIN_EMBEDDINGS_AUX
        elif split == "val":
            path_primary = Config.VAL_EMBEDDINGS_PRIMARY
            path_aux = Config.VAL_EMBEDDINGS_AUX
        elif split == "test":
            path_primary = Config.TEST_EMBEDDINGS_PRIMARY
            path_aux = Config.TEST_EMBEDDINGS_AUX
        else:
            raise ValueError(f"Unknown split: {split}")

        path_primary = self._get_cache_path(path_primary, debug)
        path_aux = self._get_cache_path(path_aux, debug)

        # View 1: Primary Backbone (MiniLM)
        emb_primary = self.embedder.get_embeddings(
            texts, Config.PRIMARY_BACKBONE, path_primary, load_cached=load_cached
        )

        # View 2: Auxiliary Backbone (MPNet)
        emb_aux = self.embedder.get_embeddings(
            texts, Config.AUX_BACKBONE, path_aux, load_cached=load_cached
        )

        # View 3: Metadata
        # Extract columns defined in Config
        meta_features = df[Config.METADATA_COLS].values.astype(np.float32)

        # Handle simple NaNs in metadata if any (though analysis showed none)
        # We fill with 0 as a safe fallback before the pipeline's QuantileTransformer
        if np.isnan(meta_features).any():
            logger.warning("NaNs detected in metadata. Filling with 0.")
            meta_features = np.nan_to_num(meta_features, nan=0.0)

        # 3. Concatenate Features
        # Structure: [Primary (384) | Aux (768) | Meta (10)]
        X_combined = np.hstack([emb_primary, emb_aux, meta_features])

        # 4. Define Slices
        dim_primary = emb_primary.shape[1]
        dim_aux = emb_aux.shape[1]
        dim_meta = meta_features.shape[1]

        slices = {
            "primary": slice(0, dim_primary),
            "aux": slice(dim_primary, dim_primary + dim_aux),
            "meta": slice(dim_primary + dim_aux, dim_primary + dim_aux + dim_meta),
        }

        logger.info(f"Data prepared for {split}. X shape: {X_combined.shape}")
        logger.info(f"Feature Slices: {slices}")

        return {"X": X_combined, "y": y, "ids": ids, "feature_slices": slices}
