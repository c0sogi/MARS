import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from library.config import SBERT_MODEL_NAME, CACHE_DIR, SEED
from library.utils import set_seed


class SBERTEmbedder:
    """
    Wrapper for Sentence-BERT models to generate dense text embeddings.
    """

    def __init__(self, model_name=SBERT_MODEL_NAME):
        """
        Initialize the SBERT model.

        Args:
            model_name (str): Name of the pre-trained model to load.
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Set seed before loading model to ensure deterministic behavior if applicable
        set_seed(SEED)
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.model.eval()

    def transform(self, texts, batch_size=32):
        """
        Generate L2-normalized embeddings for a list of texts.

        Args:
            texts (list or np.ndarray): List of text strings.
            batch_size (int): Batch size for inference.

        Returns:
            np.ndarray: L2-normalized embeddings of shape (n_samples, embedding_dim).
        """
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()

        # encode() with normalize_embeddings=True performs L2 normalization
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
        )
        return embeddings


def generate_embeddings(text_data, cache_name, load_cached_data=True, batch_size=32):
    """
    Generates embeddings for the provided text data with caching.

    Args:
        text_data (list or np.ndarray): The text data to encode.
        cache_name (str): Identifier for the cache file (e.g., 'train', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.
        batch_size (int): Batch size for the embedder.

    Returns:
        np.ndarray: The embeddings.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    file_path = os.path.join(CACHE_DIR, f"{cache_name}_embeddings.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(file_path):
        print(f"Loading {cache_name} embeddings from cache: {file_path}")
        try:
            embeddings = np.load(file_path)
            # Basic validation check
            if len(embeddings) == len(text_data):
                return embeddings
            else:
                print(
                    f"Cached embeddings size mismatch ({len(embeddings)} vs {len(text_data)}). Recomputing..."
                )
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Computing {cache_name} embeddings using {SBERT_MODEL_NAME}...")
    embedder = SBERTEmbedder()
    embeddings = embedder.transform(text_data, batch_size=batch_size)

    # 3. Save to cache
    print(f"Saving {cache_name} embeddings to cache: {file_path}")
    np.save(file_path, embeddings)

    return embeddings


def prepare_design_matrix(embeddings, numeric_data):
    """
    Concatenates text embeddings and numerical metadata into a single design matrix.

    Args:
        embeddings (np.ndarray): Text embeddings (N, D_text).
        numeric_data (np.ndarray): Numerical features (N, D_num).

    Returns:
        tuple:
            - np.ndarray: Combined feature matrix (N, D_text + D_num).
            - int: The column index where metadata features start.
    """
    # Ensure inputs are numpy arrays
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)
    if not isinstance(numeric_data, np.ndarray):
        numeric_data = np.array(numeric_data)

    # Check dimensions
    if embeddings.shape[0] != numeric_data.shape[0]:
        raise ValueError(
            f"Row mismatch: embeddings have {embeddings.shape[0]} rows, "
            f"numeric_data has {numeric_data.shape[0]} rows."
        )

    # Concatenate horizontally
    X_combined = np.hstack([embeddings, numeric_data])

    # Calculate metadata start index
    metadata_start_idx = embeddings.shape[1]

    return X_combined, metadata_start_idx
