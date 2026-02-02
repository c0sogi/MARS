import os
import random
import numpy as np
import torch
import scipy.ndimage
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_levenshtein_distance(seq1, seq2):
    """
    Computes the Levenshtein distance between two sequences.

    Args:
        seq1 (list or np.array): First sequence of items (e.g., predicted labels).
        seq2 (list or np.array): Second sequence of items (e.g., ground truth labels).

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
                    matrix[x - 1, y - 1] + 1,  # Substitution
                    matrix[x, y - 1] + 1,  # Insertion
                )
    return matrix[size_x - 1, size_y - 1]


def calculate_levenshtein_accuracy(predictions, targets):
    """
    Calculates the overall error rate based on Levenshtein distance.
    Metric = Sum(Distances) / Sum(GroundTruthLengths)

    Args:
        predictions (list of lists): Predicted sequences.
        targets (list of lists): Ground truth sequences.

    Returns:
        float: The error rate (lower is better).
    """
    total_distance = 0
    total_len = 0

    for p, t in zip(predictions, targets):
        dist = compute_levenshtein_distance(p, t)
        total_distance += dist
        total_len += len(t)

    if total_len == 0:
        return 0.0 if total_distance == 0 else float("inf")

    return total_distance / total_len


def median_filter_prediction(
    prediction_sequence, kernel_size=Config.MEDIAN_FILTER_KERNEL
):
    """
    Applies a median filter to the prediction sequence to smooth out noise.
    Uses nearest-neighbor padding to preserve boundaries.

    Args:
        prediction_sequence (np.array or list): Sequence of class indices.
        kernel_size (int): Size of the median filter window.

    Returns:
        np.array: Smoothed sequence.
    """
    # Ensure input is a numpy array
    seq = np.array(prediction_sequence)

    if len(seq) == 0:
        return seq

    # Apply median filter with nearest mode for boundary protection
    # This effectively pads the start with the first value and end with the last value
    smoothed_seq = scipy.ndimage.median_filter(seq, size=kernel_size, mode="nearest")

    return smoothed_seq


def make_pad_mask(lengths, max_len=None):
    """
    Creates a boolean mask where True indicates valid positions and False indicates padding.

    Args:
        lengths (torch.Tensor): Tensor containing the length of each sequence in the batch.
        max_len (int, optional): The maximum length. If None, max(lengths) is used.

    Returns:
        torch.Tensor: Boolean mask of shape (batch_size, max_len).
    """
    batch_size = lengths.size(0)
    if max_len is None:
        max_len = lengths.max().item()

    ids = (
        torch.arange(0, max_len, device=lengths.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    mask = ids < lengths.unsqueeze(1)

    return mask


def decode_predictions(
    frame_predictions, collapse_repeats=True, remove_background=True
):
    """
    Decodes frame-wise predictions into a list of gesture IDs.

    Args:
        frame_predictions (list or np.array): Sequence of frame-wise class indices.
        collapse_repeats (bool): Whether to collapse consecutive identical labels.
        remove_background (bool): Whether to remove the background class (assumed to be 0).

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    decoded = []
    last_pred = -1

    for pred in frame_predictions:
        pred = int(pred)

        if collapse_repeats and pred == last_pred:
            continue

        last_pred = pred

        # Assume 0 is background
        if remove_background and pred == 0:
            continue

        decoded.append(pred)

    return decoded
