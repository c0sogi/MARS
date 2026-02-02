import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from library.config import Config


class TextEncoder:
    """
    Handles the generation of dense text embeddings using pre-trained Transformer models.
    Includes caching mechanisms to avoid redundant inference.
    """

    def __init__(self):
        """
        Initialize the TextEncoder with model settings from Config.
        """
        self.model_name = Config.MODEL_NAME
        self.text_cols = Config.TEXT_COLS

    def encode(
        self, df: pd.DataFrame, cache_path: str, load_cached_data: bool = True
    ) -> np.ndarray:
        """
        Generates or loads embeddings for the provided DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing the text columns.
            cache_path (str): Path to the .npy file for caching embeddings.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            np.ndarray: A numpy array of shape (n_samples, embedding_dim) containing
                        L2-normalized embeddings.
        """
        # Ensure the directory for the cache path exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading embeddings from cache: {cache_path}")
            try:
                embeddings = np.load(cache_path)
                # Simple validation of shape consistency
                if embeddings.shape[0] == len(df):
                    return embeddings
                else:
                    print(
                        f"Cached embeddings shape {embeddings.shape} does not match DataFrame length {len(df)}. Recomputing..."
                    )
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")

        print(f"Generating embeddings for {len(df)} samples using {self.model_name}...")

        # 2. Prepare Text Input
        # Concatenate specified text columns with a space separator
        # We assume columns are already strings/filled by DataLoader, but we enforce string type here.
        text_series_list = [df[col].astype(str) for col in self.text_cols]

        # Efficient concatenation using pandas vectorization
        # e.g., "Title" + " " + "Body"
        full_texts = text_series_list[0]
        for series in text_series_list[1:]:
            full_texts = full_texts + " " + series

        texts_list = full_texts.tolist()

        # 3. Generate Embeddings
        # Initialize model
        model = SentenceTransformer(self.model_name)

        # Encode
        # batch_size=32 is generally safe for MPNet on standard GPUs
        # normalize_embeddings=True applies L2 normalization (Unit Norm)
        embeddings = model.encode(
            texts_list,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # 4. Save to Cache
        print(f"Saving embeddings to {cache_path}...")
        np.save(cache_path, embeddings)

        return embeddings
