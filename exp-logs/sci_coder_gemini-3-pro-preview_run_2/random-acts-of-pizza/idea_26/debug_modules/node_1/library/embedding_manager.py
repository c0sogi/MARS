import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from library.utils import save_npy, load_npy, ensure_directory


def get_embeddings(
    texts: list,
    model_name: str,
    cache_path: str,
    load_from_cache: bool = True,
    batch_size: int = 32,
    device: str = None,
) -> np.ndarray:
    """
    Generates sentence embeddings for a list of texts using a specified SentenceTransformer model.
    Handles caching to disk to avoid redundant computations.

    Args:
        texts (list): A list or iterable of strings to encode.
        model_name (str): The model identifier (e.g., 'sentence-transformers/all-MiniLM-L6-v2').
        cache_path (str): The file path where embeddings should be saved or loaded from (.npy format).
        load_from_cache (bool): If True, attempts to load embeddings from cache_path first.
        batch_size (int): The batch size used during inference.
        device (str, optional): The computation device ('cpu', 'cuda'). If None, auto-detects.

    Returns:
        np.ndarray: A numpy array containing the embeddings.
    """
    # Ensure the directory for the cache path exists
    ensure_directory(cache_path)

    # 1. Attempt to load from cache
    if load_from_cache and os.path.exists(cache_path):
        try:
            print(f"Loading embeddings from cache: {cache_path}")
            embeddings = load_npy(cache_path)
            # Basic validation to ensure length matches
            if len(embeddings) == len(texts):
                return embeddings
            else:
                print(
                    f"Cached embeddings count ({len(embeddings)}) does not match text count ({len(texts)}). Recomputing..."
                )
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Setup Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Generating embeddings using model '{model_name}' on {device}...")

    # 3. Load Model
    # We suppress the initialization output to keep logs clean
    model = SentenceTransformer(model_name, device=device)

    # 4. Encode Texts
    # convert_to_numpy=True returns a np.ndarray directly
    # show_progress_bar=False prevents cluttering the log
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,  # We handle normalization (L2) in the pipeline explicitly if needed
    )

    # 5. Save to Cache
    print(f"Saving generated embeddings to {cache_path}...")
    save_npy(embeddings, cache_path)

    return embeddings
