import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import save_to_cache, load_from_cache, setup_logger

# Setup logger
logger = setup_logger(
    "embedding_engine", os.path.join(Config.WORKING_DIR, "embedding_engine.log")
)


class EmbeddingEngine:
    def __init__(self):
        """
        Initializes the EmbeddingEngine.
        Detects available device (CUDA or CPU) for model execution.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"EmbeddingEngine initialized. Using device: {self.device}")

    def generate_embeddings(
        self,
        texts: list,
        model_name: str,
        cache_name: str,
        load_cached_data: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Generates embeddings for a list of texts using a specified SentenceTransformer model.
        Implements caching to avoid re-computation.

        Args:
            texts (list or np.ndarray): List of text strings to encode.
            model_name (str): The name of the SentenceTransformer model to use.
            cache_name (str): The filename (without extension) for the cache file.
            load_cached_data (bool): Whether to attempt loading from cache.
            batch_size (int): Batch size for inference.

        Returns:
            np.ndarray: A numpy array of shape (n_samples, embedding_dim).
        """
        # Construct full cache path
        cache_filename = f"{cache_name}.npy"
        cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        # 1. Try Loading from Cache
        if load_cached_data:
            embeddings = load_from_cache(cache_path)
            if embeddings is not None:
                logger.info(f"Loaded embeddings from cache: {cache_path}")
                # Validation check: ensure length matches texts
                if len(embeddings) == len(texts):
                    return embeddings
                else:
                    logger.warning(
                        f"Cached embeddings length ({len(embeddings)}) does not match "
                        f"input text length ({len(texts)}). Recomputing..."
                    )

        # 2. Compute Embeddings
        logger.info(f"Computing embeddings using model: {model_name}...")

        # Load model
        try:
            model = SentenceTransformer(model_name, device=self.device)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise e

        # Encode
        # convert_to_numpy=True is default, but explicit is better.
        # show_progress_bar=False to reduce clutter as requested.
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            device=self.device,
        )

        # 3. Save to Cache
        save_to_cache(embeddings, cache_path)
        logger.info(f"Saved embeddings to cache: {cache_path}")

        return embeddings

    def get_anchor_embeddings(
        self, texts: list, split_name: str, load_cached_data: bool = True
    ) -> np.ndarray:
        """
        Convenience wrapper to get Anchor (MiniLM) embeddings for a specific split.

        Args:
            texts: List of texts.
            split_name: 'train', 'val', or 'test'.
            load_cached_data: Whether to use cache.
        """
        cache_name = f"{split_name}_emb_anchor"
        return self.generate_embeddings(
            texts, Config.ANCHOR_MODEL_NAME, cache_name, load_cached_data
        )

    def get_aux_embeddings(
        self, texts: list, split_name: str, load_cached_data: bool = True
    ) -> np.ndarray:
        """
        Convenience wrapper to get Auxiliary (MPNet) embeddings for a specific split.

        Args:
            texts: List of texts.
            split_name: 'train', 'val', or 'test'.
            load_cached_data: Whether to use cache.
        """
        cache_name = f"{split_name}_emb_aux"
        return self.generate_embeddings(
            texts, Config.AUX_MODEL_NAME, cache_name, load_cached_data
        )
