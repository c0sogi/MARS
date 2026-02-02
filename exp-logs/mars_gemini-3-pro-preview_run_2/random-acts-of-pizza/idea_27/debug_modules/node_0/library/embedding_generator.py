import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import library.config as config


class EmbeddingService:
    def __init__(self):
        """
        Initialize the EmbeddingService.
        Sets the computation device based on availability (CUDA or CPU).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_embeddings(
        self, texts, model_name, cache_path, load_cached_data=True, batch_size=64
    ):
        """
        Generates embeddings for the provided texts using a SentenceTransformer model.
        Utilizes caching to avoid redundant computation.

        Args:
            texts (list, pd.Series, or np.ndarray): The text data to encode.
            model_name (str): The name of the SentenceTransformer model to use.
            cache_path (str): The file path where embeddings should be stored/loaded.
            load_cached_data (bool): If True, attempts to load from cache first.
            batch_size (int): Batch size for the encoding process.

        Returns:
            np.ndarray: The generated or loaded embeddings.
        """
        # Ensure the directory for the cache path exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # Attempt to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                embeddings = np.load(cache_path)
                # Simple consistency check: ensure number of embeddings matches number of texts
                if len(embeddings) == len(texts):
                    print(f"Loaded embeddings from cache: {cache_path}")
                    return embeddings
                else:
                    print(
                        f"Cache mismatch (stored: {len(embeddings)}, input: {len(texts)}). Recomputing..."
                    )
            except Exception as e:
                print(f"Error loading cache {cache_path}: {e}. Recomputing...")

        # If cache miss or load failed, compute embeddings
        print(f"Generating embeddings using {model_name}...")

        # Convert input to list if necessary (SentenceTransformer expects list of strings)
        if hasattr(texts, "tolist"):
            texts_list = texts.tolist()
        else:
            texts_list = list(texts)

        # Initialize model
        # We initialize inside the method to avoid holding multiple models in VRAM if not needed simultaneously
        model = SentenceTransformer(model_name, device=self.device)

        # Generate embeddings
        # normalize_embeddings is set to False. The strategy requires:
        # 1. MiniLM -> L2 Normalize (done in pipeline)
        # 2. MPNet -> PCA -> L2 Normalize (done in pipeline)
        # Therefore, we need raw embeddings here.
        embeddings = model.encode(
            texts_list,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        # Save to cache
        try:
            np.save(cache_path, embeddings)
            print(f"Saved embeddings to cache: {cache_path}")
        except Exception as e:
            print(f"Warning: Could not save embeddings to cache: {e}")

        return embeddings
