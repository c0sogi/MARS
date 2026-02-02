import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from library.utils import set_seed


class EmbeddingManager:
    """
    Manages the generation and caching of sentence embeddings using dual backbones.
    Implements the JBPCE strategy's embedding extraction phase.
    """

    def __init__(self, cache_dir="./working/idea_34"):
        """
        Initialize the EmbeddingManager.

        Args:
            cache_dir (str): Directory where embedding files will be stored/loaded.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        set_seed(42)

    def _compute_single_embedding(
        self, texts, model_name, cache_path, load_cached, batch_size
    ):
        """
        Helper method to compute or load embeddings for a single model.
        """
        if load_cached and os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}")
            return np.load(cache_path)

        print(f"Computing embeddings with {model_name}...")

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        model = SentenceTransformer(model_name, device=device)

        # Encode texts
        # Note: show_progress_bar is False to keep output clean as requested
        embeddings = model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )

        # Apply L2 Normalization immediately
        # This ensures inputs to the Joint PCA are balanced in magnitude
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        embeddings = embeddings / norms

        # Save to cache
        np.save(cache_path, embeddings)

        return embeddings

    def get_dual_backbone_embeddings(
        self, texts, prefix, load_cached=True, batch_size=32
    ):
        """
        Retrieves embeddings for the provided texts using both the Anchor (MiniLM)
        and Auxiliary (MPNet) backbones. Checks cache first.

        Args:
            texts (list of str): The list of text strings to encode.
            prefix (str): Prefix for the cache filename (e.g., 'train', 'test').
            load_cached (bool): Whether to attempt loading from cache.
            batch_size (int): Batch size for the embedding model inference.

        Returns:
            tuple: (embeddings_minilm, embeddings_mpnet)
                   Two numpy arrays containing the normalized embeddings.
        """
        # Define backbone model names
        model_a_name = "all-MiniLM-L6-v2"
        model_b_name = "all-mpnet-base-v2"

        # Define cache paths
        cache_path_a = os.path.join(self.cache_dir, f"{prefix}_emb_minilm.npy")
        cache_path_b = os.path.join(self.cache_dir, f"{prefix}_emb_mpnet.npy")

        # Compute or Load Backbone A (MiniLM)
        emb_a = self._compute_single_embedding(
            texts, model_a_name, cache_path_a, load_cached, batch_size
        )

        # Compute or Load Backbone B (MPNet)
        emb_b = self._compute_single_embedding(
            texts, model_b_name, cache_path_b, load_cached, batch_size
        )

        return emb_a, emb_b
