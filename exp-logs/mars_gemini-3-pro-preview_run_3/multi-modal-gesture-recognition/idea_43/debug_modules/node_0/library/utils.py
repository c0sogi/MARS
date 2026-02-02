import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


def compute_levenshtein_distance(hypothesis, reference):
    """
    Computes the Levenshtein distance between two sequences of gesture IDs.

    Args:
        hypothesis (list or np.ndarray): Predicted sequence of gesture IDs.
        reference (list or np.ndarray): Ground truth sequence of gesture IDs.

    Returns:
        int: The Levenshtein distance (edit distance).
    """
    len_hyp = len(hypothesis)
    len_ref = len(reference)

    # Initialize DP matrix
    # dp[i][j] is the distance between hypothesis[:i] and reference[:j]
    dp = np.zeros((len_hyp + 1, len_ref + 1), dtype=int)

    # Base cases
    for i in range(len_hyp + 1):
        dp[i][0] = i
    for j in range(len_ref + 1):
        dp[0][j] = j

    # Fill DP matrix
    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            if hypothesis[i - 1] == reference[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )

    return dp[len_hyp][len_ref]


def run_length_encoding(predictions):
    """
    Converts a sequence of frame-wise class IDs into Run-Length Encoded segments.

    Args:
        predictions (np.ndarray or list): Sequence of frame-wise class IDs.

    Returns:
        list of tuples: A list where each element is (class_id, length).
    """
    if len(predictions) == 0:
        return []

    predictions = np.array(predictions)
    # Find indices where values change
    # Append -1 to the end to ensure the last segment is captured if we used diff
    # Alternatively, standard iteration:

    segments = []
    if len(predictions) == 0:
        return segments

    current_val = predictions[0]
    current_len = 1

    for i in range(1, len(predictions)):
        val = predictions[i]
        if val == current_val:
            current_len += 1
        else:
            segments.append((current_val, current_len))
            current_val = val
            current_len = 1

    segments.append((current_val, current_len))
    return segments


def filter_short_segments(
    segments,
    min_length=Config.MIN_GESTURE_LENGTH,
    background_id=Config.BACKGROUND_CLASS_ID,
):
    """
    Filters out background segments and segments shorter than a minimum duration.

    Args:
        segments (list of tuples): List of (class_id, length) from RLE.
        min_length (int): Minimum duration in frames to keep a gesture.
        background_id (int): ID of the background class to remove.

    Returns:
        list: Filtered list of gesture IDs.
    """
    filtered_ids = []
    for class_id, length in segments:
        # Remove background
        if class_id == background_id:
            continue

        # Remove short segments (physical constraint)
        if length < min_length:
            continue

        filtered_ids.append(class_id)

    return filtered_ids


def decode_predictions(frame_predictions):
    """
    Wrapper function to convert frame predictions to the final list of gesture IDs.
    Applies RLE and filtering based on Config.

    Args:
        frame_predictions (np.ndarray): Array of shape (T,) containing class IDs.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    segments = run_length_encoding(frame_predictions)
    final_gestures = filter_short_segments(
        segments,
        min_length=Config.MIN_GESTURE_LENGTH,
        background_id=Config.BACKGROUND_CLASS_ID,
    )
    return final_gestures


class LogSpaceSmoothingLoss(nn.Module):
    """
    Implements a Truncated MSE loss on log-probabilities to enforce temporal smoothness.

    L = lambda * mean( min( (log(P_t) - log(P_{t+1}))^2, threshold ) )

    This penalizes rapid changes in prediction confidence between adjacent frames,
    smoothing the output boundaries.
    """

    def __init__(
        self,
        smoothing_lambda=Config.SMOOTHING_LAMBDA,
        threshold=Config.SMOOTHING_THRESHOLD,
    ):
        super(LogSpaceSmoothingLoss, self).__init__()
        self.smoothing_lambda = smoothing_lambda
        self.threshold = threshold

    def forward(self, log_probs):
        """
        Args:
            log_probs (torch.Tensor): Tensor of shape (Batch, Time, Classes).
                                      Should be output of F.log_softmax.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculate difference between adjacent frames: log(P_t) - log(P_{t+1})
        # slice [:, :-1, :] excludes last frame
        # slice [:, 1:, :] excludes first frame
        diff = log_probs[:, :-1, :] - log_probs[:, 1:, :]

        # Squared difference
        squared_diff = diff**2

        # Truncate the error (Truncated MSE)
        # We clamp the squared error to the threshold
        truncated_diff = torch.clamp(squared_diff, max=self.threshold)

        # Mean over all dimensions (Batch, Time-1, Classes)
        loss = truncated_diff.mean()

        return self.smoothing_lambda * loss
