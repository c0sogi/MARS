import random
import numpy as np
import torch
from scipy.ndimage import median_filter
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_levenshtein_distance(seq1, seq2):
    """
    Computes the Levenshtein distance between two sequences.

    Args:
        seq1 (list): First sequence (e.g., predicted gestures).
        seq2 (list): Second sequence (e.g., ground truth gestures).

    Returns:
        int: The Levenshtein distance.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=int)

    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = matrix[x - 1, y - 1]
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,  # Deletion
                    matrix[x, y - 1] + 1,  # Insertion
                    matrix[x - 1, y - 1] + 1,  # Substitution
                )
    return matrix[size_x - 1, size_y - 1]


def median_filter_predictions(predictions, kernel_size=15):
    """
    Applies a median filter to the frame-wise predictions to smooth out noise.
    Uses nearest-neighbor padding to prevent the deletion of valid gestures at sequence edges.

    Args:
        predictions (np.ndarray): 1D array of frame-wise class indices.
        kernel_size (int): Size of the median filter window.

    Returns:
        np.ndarray: Smoothed predictions.
    """
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Apply median filter with nearest padding (mode='nearest')
    smoothed = median_filter(predictions, size=kernel_size, mode="nearest")
    return smoothed


def decode_predictions(frame_predictions):
    """
    Decodes frame-wise predictions into a sequence of gesture labels.
    Collapses consecutive duplicates and removes background class (0).

    Args:
        frame_predictions (np.ndarray): 1D array of frame-wise class indices.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    if len(frame_predictions) == 0:
        return []

    # Collapse consecutive duplicates
    collapsed = [frame_predictions[0]]
    for i in range(1, len(frame_predictions)):
        if frame_predictions[i] != frame_predictions[i - 1]:
            collapsed.append(frame_predictions[i])

    # Remove background class (0)
    # Assuming background is 0 based on Config.CLASS_WEIGHTS index 0
    final_sequence = [label for label in collapsed if label != 0]

    return final_sequence
