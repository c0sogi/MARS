import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_levenshtein(seq1, seq2):
    """
    Computes the Levenshtein distance between two sequences using dynamic programming.

    Args:
        seq1 (list or iterable): The first sequence (e.g., predicted labels).
        seq2 (list or iterable): The second sequence (e.g., ground truth labels).

    Returns:
        int: The Levenshtein distance (edit distance).
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1

    # Create a matrix of size (len(seq1)+1) x (len(seq2)+1)
    matrix = np.zeros((size_x, size_y), dtype=int)

    # Initialize the first row and column
    # These represent transforming an empty prefix to the other prefix
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    # Fill the matrix
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                # If characters match, no cost is added
                matrix[x, y] = matrix[x - 1, y - 1]
            else:
                # Minimum of deletion, substitution, or insertion
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,  # Deletion
                    matrix[x - 1, y - 1] + 1,  # Substitution
                    matrix[x, y - 1] + 1,  # Insertion
                )

    # The bottom-right cell contains the Levenshtein distance
    return int(matrix[size_x - 1, size_y - 1])
