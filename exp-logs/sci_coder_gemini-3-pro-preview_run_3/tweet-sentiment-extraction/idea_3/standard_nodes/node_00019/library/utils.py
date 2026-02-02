import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def jaccard(str1: str, str2: str) -> float:
    """
    Calculates the word-level Jaccard similarity score between two strings.

    The function converts strings to lowercase and splits on whitespace to form sets of words.
    Jaccard score is defined as len(intersection) / len(union).

    Args:
        str1 (str): The first string (e.g., ground truth).
        str2 (str): The second string (e.g., prediction).

    Returns:
        float: The Jaccard similarity score between 0.0 and 1.0.
    """
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())

    c = a.intersection(b)
    union_len = len(a) + len(b) - len(c)

    # If both sets are empty, they are identical
    if union_len == 0:
        return 1.0 if len(a) == 0 and len(b) == 0 else 0.0

    return float(len(c)) / union_len


def decode_span(start_probs: np.ndarray, end_probs: np.ndarray) -> tuple:
    """
    Decodes the optimal start and end indices from predicted probabilities.

    This method maximizes the joint probability P(start, end) = P(start) * P(end)
    subject to the constraint that start_index <= end_index.

    Args:
        start_probs (np.ndarray): 1D array of probabilities for the start position. Shape: (seq_len,)
        end_probs (np.ndarray): 1D array of probabilities for the end position. Shape: (seq_len,)

    Returns:
        tuple: A tuple (best_start, best_end) containing the integer indices of the best span.
    """
    # Compute the joint probability matrix via outer product
    # score_matrix[i, j] represents P(start=i) * P(end=j)
    score_matrix = np.outer(start_probs, end_probs)

    # Create a mask to enforce the constraint: start_index <= end_index.
    # np.triu returns the upper triangle (including diagonal) where column_idx >= row_idx.
    # This corresponds to end_index >= start_index.
    mask = np.triu(np.ones_like(score_matrix))

    # Apply the mask. Since we are dealing with probabilities (>= 0),
    # multiplying by the binary mask sets invalid transitions to 0.
    masked_scores = score_matrix * mask

    # Find the flat index of the maximum value in the masked matrix
    flat_argmax = np.argmax(masked_scores)

    # Convert the flat index back to 2D coordinates (row=start, col=end)
    best_start, best_end = np.unravel_index(flat_argmax, masked_scores.shape)

    return int(best_start), int(best_end)
