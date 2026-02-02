import numpy as np
from scipy.ndimage import median_filter
from itertools import groupby
from library.config import POST_PROCESS_PARAMS


def levenshtein_distance(pred_seq, target_seq):
    """
    Calculates the Levenshtein distance between two sequences.

    Args:
        pred_seq (list): List of predicted gesture IDs.
        target_seq (list): List of ground truth gesture IDs.

    Returns:
        int: The edit distance.
    """
    n = len(pred_seq)
    m = len(target_seq)

    # Initialize DP matrix
    dp = np.zeros((n + 1, m + 1), dtype=int)

    # Base cases
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # Fill DP matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred_seq[i - 1] == target_seq[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )

    return dp[n][m]


def compute_levenshtein_score(predictions, ground_truths):
    """
    Computes the global error rate metric.
    Metric = Sum(Levenshtein Distances) / Total Number of Ground Truth Gestures.

    Args:
        predictions (list of lists): Predicted sequences.
        ground_truths (list of lists): Ground truth sequences.

    Returns:
        float: The calculated score (lower is better).
    """
    total_distance = 0
    total_gestures = 0

    for p, t in zip(predictions, ground_truths):
        dist = levenshtein_distance(p, t)
        total_distance += dist
        total_gestures += len(t)

    if total_gestures == 0:
        return 0.0

    return total_distance / total_gestures


def apply_median_filter(predictions, kernel_size=None):
    """
    Applies a median filter to smooth the predictions.
    Uses nearest-neighbor padding (mode='nearest') to preserve boundaries.

    Args:
        predictions (np.ndarray): 1D array of frame-wise class indices.
        kernel_size (int, optional): Size of the median filter window.
                                     Defaults to config.POST_PROCESS_PARAMS['median_window'].

    Returns:
        np.ndarray: Smoothed predictions.
    """
    if kernel_size is None:
        kernel_size = POST_PROCESS_PARAMS.get("median_window", 7)

    # Apply median filter with edge padding (replicating boundary values)
    # mode='nearest' in scipy.ndimage corresponds to replicating the edge pixel
    smoothed = median_filter(
        predictions,
        size=kernel_size,
        mode=POST_PROCESS_PARAMS.get("pad_mode", "nearest"),
    )

    return smoothed


def decode_predictions(frame_probs):
    """
    Decodes frame-wise probabilities into a sequence of gesture IDs.
    Pipeline: Argmax -> Median Filter -> Collapse Repeats -> Remove Background.

    Args:
        frame_probs (np.ndarray): Array of shape (T, NumClasses) containing class probabilities or logits.

    Returns:
        list: Ordered list of recognized gesture IDs (integers).
    """
    # 1. Convert probabilities to labels
    if frame_probs.ndim == 2:
        labels = np.argmax(frame_probs, axis=1)
    else:
        labels = frame_probs

    # 2. Apply Temporal Smoothing (Median Filter)
    # This removes jitter and fills small gaps
    labels = apply_median_filter(labels)

    # 3. Collapse consecutive repeated labels
    collapsed_labels = [k for k, g in groupby(labels)]

    # 4. Remove Background class (Index 0)
    # The gesture vocabulary is 1-20, 0 is background
    final_sequence = [x for x in collapsed_labels if x != 0]

    return final_sequence
