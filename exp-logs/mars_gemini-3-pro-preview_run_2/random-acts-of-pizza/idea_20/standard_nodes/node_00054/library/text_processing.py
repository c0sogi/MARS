import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from library.config import Config
from library.data_loader import load_dataset


class SBERTEncoder:
    """
    Wrapper for Sentence-BERT model to generate L2-normalized embeddings.
    """

    def __init__(self, model_name=Config.SBERT_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """
        Encodes a list or Series of text strings into embeddings.

        Args:
            texts (list or pd.Series): Input texts.

        Returns:
            np.ndarray: L2-normalized embeddings of shape (n_samples, embedding_dim).
        """
        # Ensure input is a list
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        # Generate embeddings (verbose=False to suppress progress bars)
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)

        # Apply L2 Normalization to project onto the hypersphere
        embeddings = normalize(embeddings, norm="l2")

        return embeddings


def generate_embeddings(split, load_cached_data=True):
    """
    Generates or loads SBERT embeddings for the specified dataset split.

    Args:
        split (str): One of 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load embeddings from disk.

    Returns:
        np.ndarray: The embeddings matrix.
    """
    # Resolve cache path based on split
    if split == "train":
        cache_path = Config.TRAIN_EMBEDDINGS_PATH
    elif split == "val":
        cache_path = Config.VAL_EMBEDDINGS_PATH
    elif split == "test":
        cache_path = Config.TEST_EMBEDDINGS_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading {split} embeddings from cache: {cache_path}")
            embeddings = np.load(cache_path)
            return embeddings
        except Exception as e:
            print(f"Failed to load embeddings cache: {e}. Recomputing...")

    # Compute from scratch
    print(f"Computing {split} embeddings from scratch...")

    # Load the processed dataframe (text columns are already cleaned/filled by data_loader)
    df = load_dataset(split=split, load_cached_data=True)

    # Construct the input text: Title + Space + Body
    # Using 'request_text_edit_aware' as per strategy
    title_col = "request_title"
    text_col = "request_text_edit_aware"

    # Ensure string types (redundant safety check)
    titles = df[title_col].fillna("").astype(str)
    bodies = df[text_col].fillna("").astype(str)

    combined_text = titles + " " + bodies

    # Initialize encoder and generate embeddings
    encoder = SBERTEncoder()
    embeddings = encoder.encode(combined_text)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"Saved {split} embeddings to {cache_path}")

    return embeddings
