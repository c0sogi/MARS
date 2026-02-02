import numpy as np
import random
import torch
from scipy.signal import medfilt
from itertools import groupby
from library.config import (
    SEED,
    MEDIAN_FILTER_KERNEL,
    MIN_GESTURE_LENGTH,
    BACKGROUND_LABEL,
)


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def levenshtein_distance(hyp, ref):
    """
    Computes the Levenshtein distance between two sequences (hypothesis and reference).
    This metric is used to evaluate the error rate of the recognized gesture sequences.

    Args:
        hyp (list or np.array): The predicted sequence of gesture IDs.
        ref (list or np.array): The ground truth sequence of gesture IDs.

    Returns:
        int: The edit distance (number of insertions, deletions, or substitutions).
    """
    n = len(hyp)
    m = len(ref)

    # Initialize DP matrix
    dp = np.zeros((n + 1, m + 1), dtype=int)

    # Base cases: transforming prefixes to empty string and vice versa
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j

    # Fill DP matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if hyp[i - 1] == ref[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i, j] = min(
                dp[i - 1, j] + 1,  # Deletion
                dp[i, j - 1] + 1,  # Insertion
                dp[i - 1, j - 1] + cost,  # Substitution
            )

    return dp[n, m]


def smooth_predictions(predictions, kernel_size=MEDIAN_FILTER_KERNEL):
    """
    Applies a median filter to smooth frame-wise predictions, reducing jitter.

    Args:
        predictions (np.array): Frame-wise class predictions.
        kernel_size (int): Window size for the median filter. Must be odd.

    Returns:
        np.array: Smoothed predictions.
    """
    # Ensure kernel_size is odd for medfilt
    if kernel_size % 2 == 0:
        kernel_size += 1
    return medfilt(predictions, kernel_size=kernel_size).astype(int)


def rle_decode(
    predictions, min_length=MIN_GESTURE_LENGTH, background_label=BACKGROUND_LABEL
):
    """
    Converts frame-wise predictions into a sequence of gesture labels using Run-Length Encoding logic.
    It groups consecutive identical labels and filters out the background class as well as
    segments shorter than the specified minimum length.

    Args:
        predictions (np.array or list): Frame-wise predictions.
        min_length (int): Minimum duration (in frames) to consider a gesture valid.
        background_label (int): The label ID representing background/no-gesture.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    sequence = []

    # Group consecutive identical labels
    for label, group in groupby(predictions):
        # Calculate the length of the current segment
        length = sum(1 for _ in group)

        # Filter out background and short segments
        if label != background_label and length >= min_length:
            sequence.append(int(label))

    return sequence
