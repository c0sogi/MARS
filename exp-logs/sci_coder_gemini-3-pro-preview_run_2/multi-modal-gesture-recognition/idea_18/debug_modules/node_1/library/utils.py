import numpy as np
from scipy.ndimage import median_filter
from itertools import groupby
from library.config import Config


def levenshtein_distance(hyp, ref):
    """
    Computes the Levenshtein distance between two sequences of gesture IDs.

    Args:
        hyp (list[int]): The predicted sequence of gesture IDs.
        ref (list[int]): The ground truth sequence of gesture IDs.

    Returns:
        int: The edit distance (insertions + deletions + substitutions).
    """
    if len(hyp) < len(ref):
        return levenshtein_distance(ref, hyp)

    if len(ref) == 0:
        return len(hyp)

    previous_row = range(len(ref) + 1)
    for i, c1 in enumerate(hyp):
        current_row = [i + 1]
        for j, c2 in enumerate(ref):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_levenshtein_score(predictions, targets):
    """
    Computes the global error rate based on Levenshtein distance.
    Metric = Sum(Levenshtein Distances) / Total Number of Ground Truth Gestures.

    Args:
        predictions (list[list[int]]): List of predicted gesture sequences.
        targets (list[list[int]]): List of ground truth gesture sequences.

    Returns:
        float: The calculated error rate.
    """
    total_distance = 0
    total_ref_length = 0

    for p, t in zip(predictions, targets):
        dist = levenshtein_distance(p, t)
        total_distance += dist
        total_ref_length += len(t)

    if total_ref_length == 0:
        return 0.0 if total_distance == 0 else float("inf")

    return total_distance / total_ref_length


def apply_median_filter(predictions, kernel_size=7):
    """
    Applies a median filter to the sequence of predictions to smooth out noise.
    Uses nearest-neighbor padding to preserve boundary gestures.

    Args:
        predictions (np.ndarray): 1D array of frame-wise class labels.
        kernel_size (int): Size of the median filter window.

    Returns:
        np.ndarray: Smoothed 1D array of labels.
    """
    # mode='nearest' implements Nearest-Neighbor Padding
    return median_filter(predictions, size=kernel_size, mode="nearest")


def decode_predictions(frame_probs, kernel_size=7):
    """
    Decodes frame-wise probabilities into an ordered list of gesture IDs.
    Pipeline: Argmax -> Median Filter -> Collapse Repeats -> Remove Background.

    Args:
        frame_probs (np.ndarray): Array of shape (T, NumClasses) containing probabilities or logits.
        kernel_size (int): Kernel size for the median filter.

    Returns:
        list[int]: The decoded sequence of gesture IDs.
    """
    # 1. Convert to discrete labels
    labels = np.argmax(frame_probs, axis=1)

    # 2. Apply Label-Space Smoothing (Median Filter)
    # Using nearest padding to protect boundaries
    smoothed_labels = apply_median_filter(labels, kernel_size=kernel_size)

    # 3. Collapse consecutive repeats
    collapsed = [k for k, g in groupby(smoothed_labels)]

    # 4. Remove Background Class
    # Config.BACKGROUND_CLASS_IDX is 0
    final_sequence = [int(x) for x in collapsed if x != Config.BACKGROUND_CLASS_IDX]

    return final_sequence
