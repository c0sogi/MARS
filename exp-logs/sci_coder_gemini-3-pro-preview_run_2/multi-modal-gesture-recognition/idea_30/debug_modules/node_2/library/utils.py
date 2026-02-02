import os
import random
import numpy as np
import torch
import scipy.ndimage
from library.config import RANDOM_SEED


def set_seed(seed=RANDOM_SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_levenshtein(seq1, seq2):
    """
    Computes the Levenshtein distance between two sequences.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))

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


def levenshtein_score(predictions, targets):
    """
    Computes the Levenshtein score (Error Rate) for the dataset.

    Args:
        predictions (list of list of int): Predicted gesture sequences.
        targets (list of list of int): Ground truth gesture sequences.

    Returns:
        float: The calculated score (Total Distance / Total True Gestures).
    """
    total_distance = 0
    total_len = 0

    for pred, target in zip(predictions, targets):
        dist = compute_levenshtein(pred, target)
        total_distance += dist
        total_len += len(target)

    if total_len == 0:
        return 0.0

    return total_distance / total_len


def post_process_predictions(frame_probs, filter_kernel=7):
    """
    Converts frame-wise probabilities into a sequence of gesture IDs using
    argmax, median filtering with nearest-neighbor padding, and duplicate collapsing.

    Args:
        frame_probs (np.ndarray or torch.Tensor): Shape (T, NumClasses).
        filter_kernel (int): Kernel size for the median filter.

    Returns:
        list of int: The predicted sequence of gesture IDs (excluding background).
    """
    if isinstance(frame_probs, torch.Tensor):
        frame_probs = frame_probs.detach().cpu().numpy()

    # 1. Argmax to get discrete labels
    labels = np.argmax(frame_probs, axis=1)

    # 2. Label-Space Smoothing: Median Filter with Nearest-Neighbor Padding
    # mode='nearest' replicates the edge values, protecting boundary gestures
    smoothed_labels = scipy.ndimage.median_filter(
        labels, size=filter_kernel, mode="nearest"
    )

    # 3. Decoding: Collapse duplicates and remove background (Class 0)
    final_sequence = []
    last_label = None

    for label in smoothed_labels:
        if label != last_label:
            if label != 0:  # 0 is background
                final_sequence.append(int(label))
            last_label = label

    return final_sequence
