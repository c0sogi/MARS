import numpy as np
from library.config import Config


def levenshtein_distance(hyp, ref):
    """
    Computes the Levenshtein distance between two sequences of labels.

    Args:
        hyp (list): Hypothesis sequence (predicted labels).
        ref (list): Reference sequence (ground truth labels).

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
    # dp[i][j] is distance between hyp[:i] and ref[:j]
    dp = np.zeros((n + 1, m + 1), dtype=int)

    # Base cases
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # Fill matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if hyp[i - 1] == ref[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )

    return dp[n][m]


def compute_levenshtein_score(predictions, ground_truths):
    """
    Computes the global Levenshtein score (Error Rate) for a batch of sequences.
    Score = Sum(Levenshtein Distances) / Sum(Reference Lengths)

    Args:
        predictions (list of lists): Predicted label sequences.
        ground_truths (list of lists): Ground truth label sequences.

    Returns:
        float: The calculated score.
    """
    total_dist = 0
    total_len = 0

    for p, g in zip(predictions, ground_truths):
        total_dist += levenshtein_distance(p, g)
        total_len += len(g)

    # Handle edge case where ground truth is empty (avoid division by zero)
    if total_len == 0:
        return 0.0 if total_dist == 0 else float("inf")

    return total_dist / total_len


def run_length_encoding(predictions):
    """
    Converts a frame-wise label sequence into segments.

    Args:
        predictions (np.ndarray or list): Sequence of class indices.

    Returns:
        list of dict: List of segments with 'label', 'start', 'end', 'length'.
                      'start' is inclusive, 'end' is exclusive.
    """
    if len(predictions) == 0:
        return []

    segments = []
    curr_label = predictions[0]
    curr_start = 0

    for i in range(1, len(predictions)):
        if predictions[i] != curr_label:
            segments.append(
                {
                    "label": int(curr_label),
                    "start": curr_start,
                    "end": i,
                    "length": i - curr_start,
                }
            )
            curr_label = predictions[i]
            curr_start = i

    # Append last segment
    segments.append(
        {
            "label": int(curr_label),
            "start": curr_start,
            "end": len(predictions),
            "length": len(predictions) - curr_start,
        }
    )

    return segments


def filter_short_segments(segments, min_duration=None):
    """
    Removes segments shorter than min_duration.

    Args:
        segments (list of dict): List of segments from run_length_encoding.
        min_duration (int, optional): Minimum length. Defaults to Config.MIN_GESTURE_DURATION.

    Returns:
        list of dict: Filtered segments.
    """
    if min_duration is None:
        min_duration = Config.MIN_GESTURE_DURATION

    return [s for s in segments if s["length"] >= min_duration]


def decode_predictions_to_labels(frame_predictions):
    """
    Decodes frame-wise class indices into the final sequence of gesture IDs.
    Applies RLE, length filtering, and background removal.

    Args:
        frame_predictions (np.ndarray): Array of shape (T,) with class indices.

    Returns:
        list: Ordered list of recognized gesture IDs (integers).
    """
    # 1. Run Length Encoding
    segments = run_length_encoding(frame_predictions)

    # 2. Filter short segments
    filtered_segments = filter_short_segments(segments)

    # 3. Extract non-background labels
    final_labels = []
    for seg in filtered_segments:
        label = seg["label"]
        if label != Config.BACKGROUND_CLASS_ID:
            final_labels.append(label)

    return final_labels
