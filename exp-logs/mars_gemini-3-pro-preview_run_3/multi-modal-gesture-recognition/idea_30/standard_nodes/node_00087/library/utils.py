import torch
import torch.nn as nn
import numpy as np
import itertools
from typing import List, Tuple, Union

# Import configuration for default values
from library import config


class TruncatedMSELoss(nn.Module):
    """
    Computes the Truncated Mean Squared Error (MSE) loss between adjacent frames
    in the log-probability space. This acts as a smoothing regularizer.

    Formula: Mean( min( (log_p[t] - log_p[t-1])^2, threshold^2 ) )
    """

    def __init__(self, threshold: float = config.SMOOTHING_THRESHOLD):
        super(TruncatedMSELoss, self).__init__()
        self.threshold = threshold
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_probs: Tensor of shape (Batch, Classes, Time) containing log probabilities.
        Returns:
            Scalar loss value.
        """
        # Calculate differences between adjacent time steps
        # Shape: (Batch, Classes, Time-1)
        diff = log_probs[:, :, 1:] - log_probs[:, :, :-1]

        # Compute squared differences
        squared_diff = diff**2

        # Truncate the squared errors at threshold^2
        truncated_diff = torch.clamp(squared_diff, max=self.threshold**2)

        # Return the mean loss
        return torch.mean(truncated_diff)


def levenshtein_distance(hyp: List[int], ref: List[int]) -> int:
    """
    Computes the Levenshtein distance between two sequences of integers.
    Used for the competition metric.
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

    # DP calculation
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


def compute_levenshtein_ratio(
    hypotheses: List[List[int]], references: List[List[int]]
) -> float:
    """
    Computes the global error rate metric:
    Sum(Levenshtein Distances) / Total Number of Gestures in Truth
    """
    total_distance = 0
    total_ref_gestures = 0

    for h, r in zip(hypotheses, references):
        dist = levenshtein_distance(h, r)
        total_distance += dist
        total_ref_gestures += len(r)

    if total_ref_gestures == 0:
        return 0.0 if total_distance == 0 else float("inf")

    return total_distance / total_ref_gestures


def rle_encode(predictions: Union[List[int], np.ndarray]) -> List[Tuple[int, int]]:
    """
    Run-Length Encoding for frame-wise predictions.

    Args:
        predictions: List or 1D array of class IDs.

    Returns:
        List of tuples (class_id, length).
    """
    encoded = []
    for k, g in itertools.groupby(predictions):
        length = sum(1 for _ in g)
        encoded.append((k, length))
    return encoded


def rle_decode(segments: List[Tuple[int, int]]) -> np.ndarray:
    """
    Decodes RLE segments back into a frame-wise array.

    Args:
        segments: List of tuples (class_id, length).

    Returns:
        Numpy array of frame labels.
    """
    decoded = []
    for label, length in segments:
        decoded.extend([label] * length)
    return np.array(decoded)


def filter_short_segments(
    segments: List[Tuple[int, int]], min_length: int = config.MIN_GESTURE_LENGTH
) -> List[Tuple[int, int]]:
    """
    Removes segments that are shorter than min_length.

    Args:
        segments: List of (class_id, length) tuples.
        min_length: Minimum frames required to keep a segment.

    Returns:
        Filtered list of segments.
    """
    return [seg for seg in segments if seg[1] >= min_length]


def process_predictions(
    frame_probs: np.ndarray,
    min_length: int = config.MIN_GESTURE_LENGTH,
    bg_class_idx: int = 0,
) -> List[int]:
    """
    Full post-processing pipeline:
    1. Argmax to get frame IDs.
    2. RLE Encode.
    3. Filter short segments.
    4. Remove background class.

    Args:
        frame_probs: Array of shape (Time, Classes) or (Time,) containing class IDs.
                     If (Time, Classes), argmax is applied.
        min_length: Minimum duration for a valid gesture.
        bg_class_idx: The index representing the background class (to be ignored in final output).

    Returns:
        List of recognized gesture IDs (excluding background).
    """
    # 1. Get Frame IDs
    if frame_probs.ndim == 2:
        frame_ids = np.argmax(frame_probs, axis=1)
    else:
        frame_ids = frame_probs

    # 2. RLE Encode
    segments = rle_encode(frame_ids)

    # 3. Filter Short Segments
    # We filter ALL segments based on length first to clean up noise
    filtered_segments = filter_short_segments(segments, min_length)

    # 4. Extract Non-Background Sequence
    final_sequence = [
        label for label, length in filtered_segments if label != bg_class_idx
    ]

    return final_sequence
