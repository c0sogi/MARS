import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_glove_embeddings(
    vocab: dict,
    embedding_dim: int = Config.EMBED_DIM,
    glove_path: str = None,
    load_cached_data: bool = True,
) -> np.ndarray:
    """
    Creates an embedding matrix for the given vocabulary.
    Parses a GloVe text file if provided; otherwise initializes randomly.
    Implements caching to disk.

    Args:
        vocab (dict): Dictionary mapping tokens to integer indices.
        embedding_dim (int): Dimension of the embeddings.
        glove_path (str, optional): Path to the pre-trained GloVe text file.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        np.ndarray: Embedding matrix of shape (vocab_size, embedding_dim).
    """
    cache_path = Config.EMBEDDING_MATRIX_PATH
    vocab_size = len(vocab)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # print(f"Loading embedding matrix from cache: {cache_path}")
            embedding_matrix = np.load(cache_path)

            # Verify shape integrity
            if embedding_matrix.shape == (vocab_size, embedding_dim):
                return embedding_matrix.astype(np.float32)
            else:
                print(
                    f"Cached embedding matrix shape mismatch. Expected {(vocab_size, embedding_dim)}, got {embedding_matrix.shape}. Recomputing."
                )
        except Exception as e:
            print(f"Failed to load cached embeddings: {e}. Recomputing.")

    # 2. Compute from scratch
    print("Initializing embedding matrix...")

    # Initialize with random normal distribution (standard for embeddings)
    # Using a smaller scale to keep initial gradients stable
    embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, embedding_dim))

    # Ensure PAD token is zero vector
    if Config.PAD_TOKEN in vocab:
        pad_idx = vocab[Config.PAD_TOKEN]
        embedding_matrix[pad_idx] = np.zeros(embedding_dim)

    # Parse GloVe file if provided
    if glove_path and os.path.exists(glove_path):
        print(f"Parsing GloVe embeddings from {glove_path}...")
        hits = 0
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                if word in vocab:
                    # Some GloVe lines might be malformed, check length
                    if len(parts) == embedding_dim + 1:
                        vector = np.array(parts[1:], dtype=np.float32)
                        embedding_matrix[vocab[word]] = vector
                        hits += 1

        coverage = hits / vocab_size if vocab_size > 0 else 0
        print(f"Loaded {hits} vectors from GloVe. Vocabulary coverage: {coverage:.2%}")
    else:
        print("GloVe path not provided or file not found. Using random initialization.")

    # 3. Save to cache
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embedding_matrix)
        print(f"Saved embedding matrix to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save embedding matrix to cache: {e}")

    return embedding_matrix.astype(np.float32)
