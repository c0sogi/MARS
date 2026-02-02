import os
import random
import numpy as np
import torch


def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_glove_embeddings(vocab, embedding_dim, glove_path=None):
    """
    Creates an embedding matrix for the given vocabulary. If a path to a GloVe
    file is provided and exists, it loads the pre-trained vectors. Otherwise,
    it initializes the matrix randomly.

    Args:
        vocab (dict): A dictionary mapping tokens to integer indices.
        embedding_dim (int): The dimension of the embedding vectors.
        glove_path (str, optional): Path to the GloVe text file. Defaults to None.

    Returns:
        numpy.ndarray: An embedding matrix of shape (vocab_size, embedding_dim).
    """
    vocab_size = len(vocab)
    # Initialize with random normal distribution
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim))

    # If no path provided or file doesn't exist, return random initialization
    if not glove_path or not os.path.exists(glove_path):
        print(
            f"GloVe path '{glove_path}' not found or not provided. Using random initialization."
        )
        return embedding_matrix.astype(np.float32)

    print(f"Loading GloVe embeddings from {glove_path}...")
    hits = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in vocab:
                try:
                    vector = np.asarray(values[1:], dtype="float32")
                    if len(vector) == embedding_dim:
                        embedding_matrix[vocab[word]] = vector
                        hits += 1
                except ValueError:
                    continue

    print(f"Loaded {hits} vectors out of {vocab_size} vocabulary size.")
    return embedding_matrix.astype(np.float32)


def compute_f1(pred_span, true_span):
    """
    Computes the F1 score based on token overlap between a predicted span
    and a ground truth span.

    Args:
        pred_span (tuple): A tuple (start_index, end_index) for the prediction.
                           Indices are inclusive. (-1, -1) indicates no answer.
        true_span (tuple): A tuple (start_index, end_index) for the ground truth.
                           Indices are inclusive. (-1, -1) indicates no answer.

    Returns:
        float: The F1 score (0.0 to 1.0).
    """
    pred_start, pred_end = pred_span
    true_start, true_end = true_span

    # Check for "no answer" cases
    if pred_start == -1 or pred_end == -1:
        pred_tokens = set()
    else:
        # Range is exclusive at the end, so add 1 to include the end token
        pred_tokens = set(range(pred_start, pred_end + 1))

    if true_start == -1 or true_end == -1:
        true_tokens = set()
    else:
        true_tokens = set(range(true_start, true_end + 1))

    # If both are empty, it's a perfect match (True Negative)
    if len(pred_tokens) == 0 and len(true_tokens) == 0:
        return 1.0

    # If one is empty and the other is not, F1 is 0
    if len(pred_tokens) == 0 or len(true_tokens) == 0:
        return 0.0

    common_tokens = pred_tokens.intersection(true_tokens)

    if len(common_tokens) == 0:
        return 0.0

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(true_tokens)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
