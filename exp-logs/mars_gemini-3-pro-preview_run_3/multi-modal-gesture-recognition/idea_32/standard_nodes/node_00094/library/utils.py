import numpy as np
import torch
import torch.nn as nn
from library import config


def levenshtein_distance(preds, targets):
    """
    Computes the Levenshtein distance between two sequences of gesture IDs.

    Args:
        preds (list): List of predicted gesture IDs.
        targets (list): List of ground truth gesture IDs.

    Returns:
        int: The edit distance.
    """
    n = len(preds)
    m = len(targets)

    # Initialize matrix of size (n+1) x (m+1)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    # Base cases
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # DP calculation
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if preds[i - 1] == targets[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )

    return dp[n][m]


def compute_levenshtein_score(predictions_list, targets_list):
    """
    Computes the normalized Levenshtein score (Error Rate) over a dataset.

    Args:
        predictions_list (list of lists): Predicted sequences for each sample.
        targets_list (list of lists): Ground truth sequences for each sample.

    Returns:
        float: Total distance / Total ground truth length.
    """
    total_distance = 0
    total_length = 0

    for p, t in zip(predictions_list, targets_list):
        dist = levenshtein_distance(p, t)
        total_distance += dist
        total_length += len(t)

    if total_length == 0:
        return 0.0

    return total_distance / total_length


class TruncatedMSELoss(nn.Module):
    """
    Truncated MSE Loss for Log-Space Smoothing.
    Penalizes rapid changes in log-probabilities between adjacent frames,
    clamped at a threshold to allow for genuine gesture transitions.
    """

    def __init__(self, threshold=1.0):
        super(TruncatedMSELoss, self).__init__()
        self.threshold = threshold
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, log_probs):
        """
        Args:
            log_probs (torch.Tensor): Tensor of shape (Batch, Time, Classes).

        Returns:
            torch.Tensor: Scalar loss.
        """
        # Calculate difference between t and t-1
        # Input shape: (B, T, C)
        # diff shape: (B, T-1, C)
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Squared difference
        squared_diff = diff**2

        # Truncate (clamp) the squared difference
        # We clamp the squared value to threshold^2
        truncated_diff = torch.clamp(squared_diff, max=self.threshold**2)

        # Return mean
        return torch.mean(truncated_diff)


def rle_encode(arr):
    """
    Performs Run-Length Encoding on a 1D array/list.

    Args:
        arr (list or np.array): Input sequence.

    Returns:
        list of tuples: [(value, count), ...]
    """
    if len(arr) == 0:
        return []

    segments = []
    current_val = arr[0]
    current_count = 1

    for x in arr[1:]:
        if x == current_val:
            current_count += 1
        else:
            segments.append((current_val, current_count))
            current_val = x
            current_count = 1

    segments.append((current_val, current_count))
    return segments


def process_gesture_sequence(
    frame_predictions,
    min_duration=config.MIN_GESTURE_DURATION,
    background_id=config.BACKGROUND_CLASS_ID,
):
    """
    Converts frame-wise class predictions into a clean sequence of gesture IDs.
    Applies filtering of short segments and merging of adjacent identical labels.

    Args:
        frame_predictions (np.array): 1D array of class IDs.
        min_duration (int): Minimum frames for a segment to be considered valid.
        background_id (int): Class ID for background/null gesture.

    Returns:
        list: Ordered list of recognized gesture IDs (integers).
    """
    # 1. Initial RLE
    segments = rle_encode(frame_predictions)

    # 2. Filter short segments
    # We remove any segment (background or gesture) that is too short.
    # This effectively treats short glitches as non-existent, allowing neighbors to merge.
    filtered_segments = [s for s in segments if s[1] >= min_duration]

    if not filtered_segments:
        return []

    # 3. Merge adjacent identical segments
    # (e.g., A(50), ShortNoise(2), A(50) -> A(50), A(50) -> A(100))
    merged_segments = []
    if len(filtered_segments) > 0:
        current_label, current_count = filtered_segments[0]

        for i in range(1, len(filtered_segments)):
            next_label, next_count = filtered_segments[i]
            if next_label == current_label:
                current_count += next_count
            else:
                merged_segments.append((current_label, current_count))
                current_label = next_label
                current_count = next_count
        merged_segments.append((current_label, current_count))

    # 4. Extract final gesture sequence (remove background)
    final_sequence = [
        label for label, count in merged_segments if label != background_id
    ]

    return final_sequence
