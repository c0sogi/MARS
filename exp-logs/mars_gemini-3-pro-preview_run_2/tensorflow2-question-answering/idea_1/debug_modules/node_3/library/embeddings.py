import os
import numpy as np
from library.config import Config


def load_glove_embeddings(path):
    """
    Parses the GloVe text file into a dictionary mapping words to vectors.

    Args:
        path (str): Path to the GloVe text file.

    Returns:
        dict: A dictionary where keys are words and values are numpy arrays.
    """
    embeddings_index = {}
    if path is None or not os.path.exists(path):
        print(
            f"GloVe path not provided or file not found: {path}. Proceeding with random initialization for all tokens."
        )
        return embeddings_index

    print(f"Parsing GloVe embeddings from {path}...")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    coefs = np.asarray(values[1:], dtype="float32")
                    embeddings_index[word] = coefs
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error reading GloVe file: {e}")

    print(f"Loaded {len(embeddings_index)} word vectors.")
    return embeddings_index


def create_embedding_matrix(
    tokenizer,
    glove_path=None,
    embedding_dim=Config.EMBEDDING_DIM,
    load_cached_data=True,
):
    """
    Constructs a weight matrix that maps the Tokenizer's vocabulary to dense vectors.
    Implements caching to avoid re-parsing GloVe and re-building the matrix.

    Args:
        tokenizer: An instance of library.data_utils.Tokenizer.
        glove_path (str): Path to the pre-trained GloVe file.
        embedding_dim (int): Dimension of the embeddings.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        np.array: Embedding matrix of shape (vocab_size, embedding_dim).
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = Config.EMBEDDING_MATRIX_PATH
    vocab_size = tokenizer.vocab_size

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading embedding matrix from {cache_path}...")
        try:
            embedding_matrix = np.load(cache_path)
            if embedding_matrix.shape == (vocab_size, embedding_dim):
                return embedding_matrix
            else:
                print(
                    f"Cached matrix shape {embedding_matrix.shape} does not match expected {(vocab_size, embedding_dim)}. Recomputing..."
                )
        except Exception as e:
            print(f"Failed to load cached embedding matrix: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Creating embedding matrix from scratch...")

    # Load GloVe vectors
    embeddings_index = load_glove_embeddings(glove_path)

    # Initialize matrix with random normal distribution for unknown words
    # Scale initialization to keep variance reasonable (Xavier-like)
    scale = 1.0 / np.sqrt(embedding_dim)
    embedding_matrix = np.random.normal(
        loc=0.0, scale=scale, size=(vocab_size, embedding_dim)
    ).astype(np.float32)

    # Explicitly set PAD token to zeros
    if hasattr(tokenizer, "word2idx") and Config.PAD_TOKEN in tokenizer.word2idx:
        pad_idx = tokenizer.word2idx[Config.PAD_TOKEN]
        embedding_matrix[pad_idx] = np.zeros(embedding_dim)

    hits = 0
    misses = 0

    # Map tokens to vectors
    for word, i in tokenizer.word2idx.items():
        if i >= vocab_size:
            continue

        # Skip PAD token as it is already zeroed
        if word == Config.PAD_TOKEN:
            continue

        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            if len(embedding_vector) == embedding_dim:
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                # Dimension mismatch (e.g. loaded 50d file for 100d config)
                # Keep random initialization
                misses += 1
        else:
            misses += 1

    print(f"Embedding matrix built. Hits: {hits}, Misses: {misses}")

    # 3. Save to cache
    print(f"Saving embedding matrix to {cache_path}...")
    try:
        np.save(cache_path, embedding_matrix)
    except Exception as e:
        print(f"Failed to save embedding matrix to cache: {e}")

    return embedding_matrix
