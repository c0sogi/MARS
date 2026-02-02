import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config


class EmbeddingEngine:
    """
    Handles the conversion of text data into dense vector embeddings using
    pre-trained Sentence Transformer models.
    """

    def __init__(self):
        """
        Initializes the SentenceTransformer model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing EmbeddingEngine on device: {self.device}")

        # Load the pre-trained model
        # We use the model name defined in the configuration
        self.model = SentenceTransformer(Config.MODEL_NAME, device=self.device)

        # Set the maximum sequence length
        self.model.max_seq_length = Config.MAX_LENGTH

    def generate_embeddings(self, texts, data_type, load_cached_data=True):
        """
        Generates embeddings for a list of texts, with caching support.

        Args:
            texts (list of str): The list of essay texts to encode.
            data_type (str): The type of data ('train', 'val', 'test') for cache naming.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            np.ndarray: A numpy array of shape (n_samples, embedding_dim).
        """
        # Ensure the cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Construct cache filename
        # We append '_debug' if running in debug mode to avoid polluting the main cache
        suffix = "_debug" if Config.DEBUG else ""
        filename = f"{data_type}_embeddings{suffix}.npy"
        cache_path = os.path.join(Config.CACHE_DIR, filename)

        # 1. Try loading from cache
        if load_cached_data:
            if os.path.exists(cache_path):
                try:
                    print(f"Loading embeddings from cache: {cache_path}")
                    embeddings = np.load(cache_path)

                    # Verify length matches input
                    if len(embeddings) == len(texts):
                        return embeddings
                    else:
                        print(
                            f"Cache size mismatch ({len(embeddings)} vs {len(texts)}). Recomputing."
                        )
                except Exception as e:
                    print(f"Failed to load cache ({e}). Recomputing.")
            else:
                print(f"Cache not found for {data_type}. Computing from scratch.")

        # 2. Compute embeddings
        print(f"Generating embeddings for {len(texts)} texts...")

        # encode() handles batching internally.
        # convert_to_numpy=True returns a numpy array directly.
        # normalize_embeddings=True ensures unit length, often helpful for linear models.
        embeddings = self.model.encode(
            texts,
            batch_size=Config.BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
        )

        # 3. Save to cache
        print(f"Saving embeddings to cache: {cache_path}")
        try:
            np.save(cache_path, embeddings)
        except Exception as e:
            print(f"Warning: Failed to save cache to {cache_path}. Error: {e}")

        return embeddings
