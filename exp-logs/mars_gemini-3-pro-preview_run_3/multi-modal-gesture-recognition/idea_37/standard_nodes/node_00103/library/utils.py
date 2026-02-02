import os
import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def levenshtein_distance(hypothesis, reference):
    """
    Computes the Levenshtein distance between two sequences.

    Args:
        hypothesis (list): The predicted sequence of labels.
        reference (list): The ground truth sequence of labels.

    Returns:
        int: The edit distance.
    """
    len_hyp = len(hypothesis)
    len_ref = len(reference)

    # Initialize matrix of size (len_hyp + 1) x (len_ref + 1)
    # dp[i][j] stores distance between hypothesis[:i] and reference[:j]
    dp = np.zeros((len_hyp + 1, len_ref + 1), dtype=int)

    for i in range(len_hyp + 1):
        dp[i][0] = i
    for j in range(len_ref + 1):
        dp[0][j] = j

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


def run_length_encoding(predictions, min_duration=5, background_class=0):
    """
    Converts frame-wise predictions into a sequence of gesture labels using
    Run-Length Encoding (RLE) and filters out short segments.

    Args:
        predictions (list or np.array): Frame-wise class predictions.
        min_duration (int): Minimum duration (in frames) for a segment to be kept.
        background_class (int): The class ID representing background/no-gesture.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    if len(predictions) == 0:
        return []

    # 1. Compute RLE
    segments = []
    if len(predictions) > 0:
        current_label = predictions[0]
        current_count = 1

        for i in range(1, len(predictions)):
            label = predictions[i]
            if label == current_label:
                current_count += 1
            else:
                segments.append((current_label, current_count))
                current_label = label
                current_count = 1
        # Append the last segment
        segments.append((current_label, current_count))

    # 2. Filter by duration and remove background
    final_sequence = []
    for label, count in segments:
        if count >= min_duration:
            if label != background_class:
                final_sequence.append(int(label))

    return final_sequence


class TruncatedMSELoss(nn.Module):
    """
    Implements a Truncated Mean Squared Error loss for temporal smoothing.

    Calculates the MSE between adjacent time steps in log-space, clipped at a threshold.
    L = mean( min( || log(P_t) - log(P_{t-1}) ||^2, threshold ) )

    This encourages smoothness while allowing for sharp transitions (edges)
    where the change is significant.
    """

    def __init__(self, threshold=1.0):
        """
        Args:
            threshold (float): The maximum squared error value allowed before truncation.
        """
        super(TruncatedMSELoss, self).__init__()
        self.threshold = threshold

    def forward(self, log_probs):
        """
        Args:
            log_probs (torch.Tensor): Tensor of shape (Batch, Time, Classes).
                                      Should be log-probabilities (e.g., output of LogSoftmax).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Compute difference between adjacent frames: P_t - P_{t-1}
        # Shape: (Batch, Time-1, Classes)
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Squared difference
        sq_diff = diff**2

        # Truncate (clamp) the squared difference
        truncated_diff = torch.clamp(sq_diff, max=self.threshold)

        # Mean over all dimensions
        loss = truncated_diff.mean()

        return loss
