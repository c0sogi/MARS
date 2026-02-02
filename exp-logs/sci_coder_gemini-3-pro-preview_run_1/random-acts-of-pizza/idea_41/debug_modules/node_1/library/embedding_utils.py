import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from library.config import SBERT_MODEL_NAME, WORKING_DIR, DEVICE


class SBERTEmbedder:
    """
    Wrapper for Sentence-BERT models to generate semantic embeddings.
    Includes caching mechanisms to persist embeddings to disk.
    """

    def __init__(self, model_name=SBERT_MODEL_NAME, device=DEVICE):
        """
        Initialize the SBERT model.

        Args:
            model_name (str): Name of the pre-trained model to load.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.model_name = model_name
        self.device = device

        # Initialize model
        # We suppress the progress bar from sentence_transformers if possible or just rely on standard output
        self.model = SentenceTransformer(model_name, device=device)
        self.model.eval()

    def encode_batch(
        self,
        texts,
        cache_key=None,
        load_cached_data=True,
        batch_size=32,
        show_progress_bar=False,
    ):
        """
        Generates embeddings for a batch of texts.

        Args:
            texts (list or pd.Series): List of text strings to encode.
            cache_key (str, optional): Unique identifier for caching. If None, caching is disabled.
                                       The file will be saved as {WORKING_DIR}/{cache_key}.npy.
            load_cached_data (bool): If True, attempts to load from cache first.
            batch_size (int): Batch size for encoding.
            show_progress_bar (bool): Whether to show a progress bar during encoding.

        Returns:
            np.ndarray: Array of embeddings with shape (n_samples, embedding_dim).
        """
        # Ensure working directory exists
        os.makedirs(WORKING_DIR, exist_ok=True)

        # 1. Handle Caching Logic
        if cache_key:
            cache_path = os.path.join(WORKING_DIR, f"{cache_key}.npy")

            if load_cached_data:
                if os.path.exists(cache_path):
                    print(f"Loading embeddings from cache: {cache_path}")
                    try:
                        embeddings = np.load(cache_path)
                        # specific check to ensure length matches
                        if len(embeddings) == len(texts):
                            return embeddings
                        else:
                            print(
                                f"Cached embeddings length ({len(embeddings)}) mismatch with input texts ({len(texts)}). Recomputing."
                            )
                    except Exception as e:
                        print(f"Failed to load embedding cache: {e}. Recomputing.")
                else:
                    print(f"Cache file not found: {cache_path}. Computing...")
            else:
                print(f"Ignoring cache for {cache_key}. Computing...")

        # 2. Preprocessing
        # Ensure texts are strings and handle potential NaNs
        cleaned_texts = [
            str(t) if t is not None and not isinstance(t, float) else "" for t in texts
        ]
        # Handle case where float NaN might have slipped in as a float type in a list
        cleaned_texts = [t if t != "nan" else "" for t in cleaned_texts]

        # 3. Computation
        # sentence-transformers encode returns a numpy array by default when convert_to_tensor=False
        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            device=self.device,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity usage later
        )

        # 4. Save to Cache
        if cache_key:
            try:
                np.save(cache_path, embeddings)
                print(f"Saved embeddings to cache: {cache_path}")
            except Exception as e:
                print(f"Warning: Could not save embeddings to cache: {e}")

        return embeddings
