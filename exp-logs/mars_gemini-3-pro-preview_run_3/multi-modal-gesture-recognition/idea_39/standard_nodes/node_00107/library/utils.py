import numpy as np
import torch
import torch.nn as nn
from library.config import BACKGROUND_CLASS_ID


def levenshtein_distance(hyp, ref):
    """
    Computes the Levenshtein distance between two sequences.

    Args:
        hyp (list): List of predicted gesture IDs.
        ref (list): List of ground truth gesture IDs.

    Returns:
        int: The edit distance.
    """
    n = len(hyp)
    m = len(ref)

    # Initialize matrix of size (n+1) x (m+1)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    # Base cases
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if hyp[i - 1] == ref[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )

    return dp[n][m]


def compute_levenshtein_score(hypotheses, references):
    """
    Computes the global Levenshtein score (Error Rate).

    Metric: Sum of Levenshtein distances / Total number of ground truth gestures.

    Args:
        hypotheses (list of lists): Predicted sequences.
        references (list of lists): Ground truth sequences.

    Returns:
        float: The computed score.
    """
    total_distance = 0
    total_ref_length = 0

    for h, r in zip(hypotheses, references):
        dist = levenshtein_distance(h, r)
        total_distance += dist
        total_ref_length += len(r)

    if total_ref_length == 0:
        return 0.0

    return total_distance / total_ref_length


def run_length_encoding(predictions, min_duration=5):
    """
    Decodes frame-wise predictions into a sequence of gesture IDs using Run-Length Encoding.
    Filters out background class and short segments.

    Args:
        predictions (np.ndarray or list): Array of frame-wise class indices.
        min_duration (int): Minimum duration in frames for a gesture to be considered valid.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    if len(predictions) == 0:
        return []

    # Group consecutive identical values
    decoded_sequence = []

    current_label = predictions[0]
    current_length = 1

    # Iterate from second frame to end
    for i in range(1, len(predictions)):
        label = predictions[i]
        if label == current_label:
            current_length += 1
        else:
            # End of a segment
            if current_label != BACKGROUND_CLASS_ID and current_length >= min_duration:
                decoded_sequence.append(int(current_label))

            current_label = label
            current_length = 1

    # Handle the last segment
    if current_label != BACKGROUND_CLASS_ID and current_length >= min_duration:
        decoded_sequence.append(int(current_label))

    return decoded_sequence


def log_space_smoothing_loss(log_probs, threshold=1.0):
    """
    Calculates the Truncated MSE loss on the log-probabilities of adjacent frames
    to enforce temporal smoothness.

    Loss = mean( clamp( (log_P[t] - log_P[t-1])^2, max=threshold ) )

    Args:
        log_probs (torch.Tensor): Tensor of shape (Batch, Time, Classes).
        threshold (float): Maximum value for the squared difference (truncation threshold).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Calculate difference between adjacent frames along the time dimension (dim=1)
    # log_probs shape: [B, T, C]
    diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

    # Square the differences
    squared_diff = diff**2

    # Apply truncation (clamping)
    truncated_diff = torch.clamp(squared_diff, max=threshold)

    # Return mean loss
    return truncated_diff.mean()
