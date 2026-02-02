import torch
import torch.nn as nn
import numpy as np
from library.config import Config


def levenshtein_distance(p, y):
    """
    Computes the Levenshtein distance between two sequences.

    Args:
        p (list): Predicted sequence of gesture IDs.
        y (list): Target sequence of gesture IDs.

    Returns:
        int: The edit distance.
    """
    n = len(p)
    m = len(y)

    # Initialize matrix
    d = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n + 1):
        d[i, 0] = i
    for j in range(m + 1):
        d[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if p[i - 1] == y[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i, j] = min(
                d[i - 1, j] + 1,  # Deletion
                d[i, j - 1] + 1,  # Insertion
                d[i - 1, j - 1] + cost,
            )  # Substitution

    return d[n, m]


def compute_levenshtein_score(predictions, targets):
    """
    Computes the competition metric: Total Levenshtein Distance / Total True Gestures.

    Args:
        predictions (list of lists): List of predicted gesture sequences.
        targets (list of lists): List of ground truth gesture sequences.

    Returns:
        float: The error rate.
    """
    total_distance = 0
    total_true_gestures = 0

    for p, y in zip(predictions, targets):
        dist = levenshtein_distance(p, y)
        total_distance += dist
        total_true_gestures += len(y)

    if total_true_gestures == 0:
        return 0.0

    return total_distance / total_true_gestures


class TruncatedMSELoss(nn.Module):
    """
    Log-Space Smoothing Loss: Truncated MSE applied to adjacent frames.
    Penalizes rapid fluctuations in predictions unless the change is significant
    (likely a true transition), in which case the truncation limits the penalty.
    """

    def __init__(self, threshold=Config.SMOOTHING_THRESHOLD):
        super(TruncatedMSELoss, self).__init__()
        self.threshold_sq = threshold**2

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Log-probabilities of shape (Batch, Time, Classes)
                              or (Batch, Classes, Time).
        """
        # Ensure shape is (Batch, Time, Classes) for temporal diff
        if x.dim() == 3 and x.size(1) == Config.NUM_CLASSES:
            # Permute if (Batch, Classes, Time) -> (Batch, Time, Classes)
            x = x.permute(0, 2, 1)

        # Compute difference between t and t-1
        # x[:, 1:, :] are frames 1..T
        # x[:, :-1, :] are frames 0..T-1
        diff = x[:, 1:, :] - x[:, :-1, :]

        # Squared difference
        diff_sq = diff**2

        # Truncate (clamp) the squared error
        # We want min(diff^2, threshold^2)
        truncated_diff_sq = torch.clamp(diff_sq, max=self.threshold_sq)

        # Mean over all dimensions
        loss = torch.mean(truncated_diff_sq)

        return loss


def run_length_encoding(frame_indices):
    """
    Converts a sequence of frame-wise class indices into segments.

    Args:
        frame_indices (list or np.array): Sequence of class IDs.

    Returns:
        list of tuples: [(label, start_frame, end_frame), ...]
    """
    if len(frame_indices) == 0:
        return []

    segments = []
    current_label = frame_indices[0]
    start_frame = 0

    for i in range(1, len(frame_indices)):
        if frame_indices[i] != current_label:
            segments.append((current_label, start_frame, i - 1))
            current_label = frame_indices[i]
            start_frame = i

    # Add the last segment
    segments.append((current_label, start_frame, len(frame_indices) - 1))

    return segments


def filter_segments(segments, min_duration=Config.MIN_GESTURE_DURATION):
    """
    Removes segments shorter than min_duration.

    Args:
        segments (list of tuples): [(label, start, end), ...]
        min_duration (int): Minimum frames required.

    Returns:
        list of tuples: Filtered segments.
    """
    filtered = []
    for label, start, end in segments:
        duration = end - start + 1
        if duration >= min_duration:
            filtered.append((label, start, end))
    return filtered


def decode_sequence(frame_probs):
    """
    Decodes frame probabilities into a sequence of gesture IDs.
    Applies RLE, duration filtering, and background removal.

    Args:
        frame_probs (np.array or torch.Tensor): Shape (Time, Classes)

    Returns:
        list: Sequence of gesture IDs (integers).
    """
    if isinstance(frame_probs, torch.Tensor):
        frame_probs = frame_probs.detach().cpu().numpy()

    # 1. Argmax to get class indices
    frame_indices = np.argmax(frame_probs, axis=-1)

    # 2. Run-Length Encoding
    segments = run_length_encoding(frame_indices)

    # 3. Filter Short Segments
    # Note: In a more complex decoder, we might merge short segments into neighbors.
    # Here we simply drop them, effectively treating them as noise/background
    # if they don't meet the physical constraint.
    valid_segments = filter_segments(segments, min_duration=Config.MIN_GESTURE_DURATION)

    # 4. Extract Gesture IDs (Remove Background Class 0)
    gesture_sequence = []
    for label, _, _ in valid_segments:
        if label != 0:
            gesture_sequence.append(int(label))

    return gesture_sequence
