import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from library.utils import setup_logger, set_seed


class EmbeddingEngine:
    def __init__(self, cache_dir="./working/idea_33"):
        """
        Initialize the EmbeddingEngine.

        Args:
            cache_dir (str): Directory to store cached embedding files.
        """
        self.logger = setup_logger("EmbeddingEngine")
        self.cache_dir = cache_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        set_seed(42)

    def generate_embeddings(
        self, texts, model_name, save_name, load_cached_data=True, batch_size=32
    ):
        """
        Generate embeddings for a list of texts using a specified SentenceTransformer model.
        Handles caching via .npy files.

        Args:
            texts (list or np.array): List of text strings to encode.
            model_name (str): Name of the SentenceTransformer model (e.g., 'all-MiniLM-L6-v2').
            save_name (str): Filename identifier for the cache (e.g., 'train_anchor').
            load_cached_data (bool): If True, attempt to load from cache first.
            batch_size (int): Batch size for encoding.

        Returns:
            np.ndarray: Matrix of embeddings.
        """
        file_path = os.path.join(self.cache_dir, f"{save_name}.npy")

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(file_path):
            self.logger.info(f"Loading cached embeddings from {file_path}...")
            try:
                embeddings = np.load(file_path)
                # Verify shape matches text count
                if len(texts) > 0 and embeddings.shape[0] != len(texts):
                    self.logger.warning(
                        f"Cached embeddings shape {embeddings.shape} does not match input length {len(texts)}. Recomputing..."
                    )
                else:
                    return embeddings
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        self.logger.info(f"Computing embeddings using {model_name} on {self.device}...")

        # Load model
        model = SentenceTransformer(model_name, device=self.device)
        model.eval()

        # Encode
        # convert to list if it's a pandas series or numpy array
        if not isinstance(texts, list):
            texts = list(texts)

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # We handle normalization later in the pipeline if needed
        )

        # 3. Save to cache
        self.logger.info(f"Saving embeddings to {file_path}...")
        np.save(file_path, embeddings)

        return embeddings

    def get_anchor_embeddings(self, texts, split_name, load_cached_data=True):
        """
        Wrapper to get embeddings from the Anchor backbone (all-MiniLM-L6-v2).

        Args:
            texts: List of texts.
            split_name: 'train', 'val', or 'test'.
            load_cached_data: Boolean.
        """
        return self.generate_embeddings(
            texts=texts,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            save_name=f"{split_name}_anchor_embeddings",
            load_cached_data=load_cached_data,
        )

    def get_auxiliary_embeddings(self, texts, split_name, load_cached_data=True):
        """
        Wrapper to get embeddings from the Auxiliary backbone (all-mpnet-base-v2).

        Args:
            texts: List of texts.
            split_name: 'train', 'val', or 'test'.
            load_cached_data: Boolean.
        """
        return self.generate_embeddings(
            texts=texts,
            model_name="sentence-transformers/all-mpnet-base-v2",
            save_name=f"{split_name}_aux_embeddings",
            load_cached_data=load_cached_data,
        )
