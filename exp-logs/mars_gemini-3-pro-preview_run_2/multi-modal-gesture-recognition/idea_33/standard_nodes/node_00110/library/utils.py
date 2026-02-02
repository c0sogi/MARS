import numpy as np
from scipy.ndimage import median_filter
from itertools import groupby


def levenshtein_distance(hyp, ref):
    """
    Computes the Levenshtein distance between two sequences (hypothesis and reference).
    Uses the standard dynamic programming approach.

    Args:
        hyp (list): The predicted sequence of gesture IDs.
        ref (list): The ground truth sequence of gesture IDs.

    Returns:
        int: The edit distance (insertions + deletions + substitutions).
    """
    n = len(hyp)
    m = len(ref)

    if n == 0:
        return m
    if m == 0:
        return n

    # Initialize DP matrix
    # matrix[i, j] is distance between hyp[:i] and ref[:j]
    matrix = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n + 1):
        matrix[i, 0] = i
    for j in range(m + 1):
        matrix[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if hyp[i - 1] == ref[j - 1] else 1
            matrix[i, j] = min(
                matrix[i - 1, j] + 1,  # Deletion
                matrix[i, j - 1] + 1,  # Insertion
                matrix[i - 1, j - 1] + cost,  # Substitution
            )

    return matrix[n, m]


def compute_normalized_levenshtein(predictions, ground_truths):
    """
    Computes the competition metric: Total Levenshtein Distance / Total Ground Truth Gestures.

    Args:
        predictions (list of lists): List containing the predicted gesture sequences for each sample.
        ground_truths (list of lists): List containing the ground truth gesture sequences for each sample.

    Returns:
        float: The calculated error rate.
    """
    total_distance = 0
    total_gt_gestures = 0

    for p, g in zip(predictions, ground_truths):
        dist = levenshtein_distance(p, g)
        total_distance += dist
        total_gt_gestures += len(g)

    if total_gt_gestures == 0:
        # Avoid division by zero; if there were no gestures and distance is 0, error is 0.
        # If distance > 0 (hallucinations), error is infinite/undefined, returning a high value or handling appropriately.
        return 0.0 if total_distance == 0 else float("inf")

    return total_distance / total_gt_gestures


def apply_median_filter(predictions, kernel_size=7):
    """
    Applies a median filter to the frame-wise predictions to smooth out noise and fill gaps.
    Uses nearest-neighbor padding (mode='nearest') to preserve predictions at the sequence boundaries.

    Args:
        predictions (np.ndarray or list): Frame-wise class indices.
        kernel_size (int): The size of the sliding window. Should be an odd number.

    Returns:
        np.ndarray: The smoothed frame-wise predictions.
    """
    predictions = np.array(predictions)
    if len(predictions) == 0:
        return predictions

    # mode='nearest' repeats the edge value, preventing boundary erosion
    smoothed = median_filter(predictions, size=kernel_size, mode="nearest")
    return smoothed


def decode_sequence(frame_predictions, background_class_id=0):
    """
    Decodes a sequence of frame-wise predictions into an ordered list of gesture IDs.
    Collapses consecutive duplicates and removes the background class.

    Args:
        frame_predictions (list or np.ndarray): Frame-wise class indices.
        background_class_id (int): The class ID representing background/null.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    # Collapse consecutive duplicates using itertools.groupby
    # x[0] gets the key (the class ID) from the group
    collapsed = [x[0] for x in groupby(frame_predictions)]

    # Remove background class instances
    decoded = [x for x in collapsed if x != background_class_id]

    return decoded


def post_process_and_decode(frame_probs, kernel_size=7, background_class_id=0):
    """
    A convenience wrapper that performs the full inference post-processing pipeline:
    Argmax -> Median Filter -> Decode Sequence.

    Args:
        frame_probs (np.ndarray): (T, C) array of class probabilities or logits, or (T,) array of indices.
        kernel_size (int): Window size for the median filter.
        background_class_id (int): ID for the background class.

    Returns:
        list: The final predicted sequence of gesture IDs.
    """
    # Convert probabilities to class indices if necessary
    if isinstance(frame_probs, np.ndarray) and frame_probs.ndim == 2:
        preds = np.argmax(frame_probs, axis=1)
    else:
        preds = np.array(frame_probs)

    # Apply smoothing
    smoothed_preds = apply_median_filter(preds, kernel_size=kernel_size)

    # Decode to gesture list
    sequence = decode_sequence(smoothed_preds, background_class_id=background_class_id)

    return sequence
