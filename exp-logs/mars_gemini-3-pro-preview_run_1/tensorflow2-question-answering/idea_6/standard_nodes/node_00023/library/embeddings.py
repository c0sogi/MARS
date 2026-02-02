import os
import numpy as np
from library.config import Config


def load_glove_embeddings(glove_path):
    """
    Parses a GloVe embedding text file into a dictionary.

    Args:
        glove_path (str): Path to the GloVe text file.

    Returns:
        dict: Mapping from word (str) to embedding vector (np.array).
    """
    embeddings_index = {}
    print(f"Attempting to load GloVe embeddings from {glove_path}...")

    try:
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    coefs = np.asarray(values[1:], dtype="float32")
                    embeddings_index[word] = coefs
                except ValueError:
                    continue
        print(f"Loaded {len(embeddings_index)} word vectors.")
    except FileNotFoundError:
        print(f"GloVe file not found at {glove_path}. Skipping load.")

    return embeddings_index


def create_embedding_matrix(word2idx, embedding_dict=None):
    """
    Creates a static embedding matrix based on the vocabulary.

    Args:
        word2idx (dict): Vocabulary mapping word -> index.
        embedding_dict (dict, optional): Pre-trained embeddings.

    Returns:
        np.ndarray: Embedding matrix of shape (vocab_size, embed_dim).
    """
    vocab_size = len(word2idx)
    embed_dim = Config.EMBED_DIM

    # Initialize with random values (scaled for stability)
    # Using a fixed seed for reproducibility within this function scope is good practice,
    # though the global seed should handle it.
    np.random.seed(Config.SEED)
    embedding_matrix = np.random.normal(
        scale=1.0 / np.sqrt(embed_dim), size=(vocab_size, embed_dim)
    )

    # Identify special tokens
    pad_idx = word2idx.get(Config.PAD_TOKEN, 0)

    # Zero out the padding token
    embedding_matrix[pad_idx] = np.zeros(embed_dim)

    hits = 0
    misses = 0

    if embedding_dict:
        for word, i in word2idx.items():
            if i >= vocab_size:
                continue

            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                # Ensure dimension matches
                if len(embedding_vector) == embed_dim:
                    embedding_matrix[i] = embedding_vector
                    hits += 1
                else:
                    # If dimension mismatch (e.g. loading 100d glove into 64d config),
                    # we keep random init. Ideally Config should match GloVe dim.
                    misses += 1
            else:
                misses += 1

        print(f"Embedding Matrix Stats: Hits={hits}, Misses={misses}")
    else:
        print("No pre-trained dictionary provided. Using random initialization.")

    return embedding_matrix.astype(np.float32)


def get_embedding_matrix(word2idx, load_cached_data=True):
    """
    Main function to retrieve the embedding matrix.
    Implements caching logic.

    Args:
        word2idx (dict): Vocabulary mapping.
        load_cached_data (bool): Whether to try loading from disk.

    Returns:
        np.ndarray: The embedding matrix.
    """
    # Ensure working directory exists
    Config.ensure_directories()

    cache_path = os.path.join(Config.WORKING_DIR, "embedding_matrix.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading embedding matrix from cache: {cache_path}")
        try:
            embedding_matrix = np.load(cache_path)
            # Verify shape matches current config/vocab
            if embedding_matrix.shape == (len(word2idx), Config.EMBED_DIM):
                return embedding_matrix
            else:
                print(
                    f"Cached matrix shape {embedding_matrix.shape} mismatch with config "
                    f"({len(word2idx)}, {Config.EMBED_DIM}). Recomputing."
                )
        except Exception as e:
            print(f"Failed to load embedding cache: {e}. Recomputing.")

    # 2. Compute from scratch
    print("Generating embedding matrix...")

    # Define a hypothetical path for GloVe.
    # In a real scenario, this would be provided.
    # Here we check, but expect to fall back to random if missing.
    glove_filename = f"glove.6B.{Config.EMBED_DIM}d.txt"
    glove_path = os.path.join(Config.INPUT_DIR, glove_filename)

    embedding_dict = load_glove_embeddings(glove_path)

    # Create matrix
    embedding_matrix = create_embedding_matrix(word2idx, embedding_dict)

    # 3. Save to cache
    try:
        np.save(cache_path, embedding_matrix)
        print(f"Saved embedding matrix to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save embedding matrix to cache: {e}")

    return embedding_matrix
