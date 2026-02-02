import os
import random
import numpy as np
import torch
import torch.nn as nn
import nltk
from library.config import TrainConfig, LABEL_MAP


# ==========================================
# Reproducibility
# ==========================================
def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# Metrics
# ==========================================
def compute_levenshtein(predicted_seqs, target_seqs):
    """
    Computes the normalized Levenshtein distance (Error Rate).

    Args:
        predicted_seqs (list of list of int): List of predicted gesture IDs for each sample.
        target_seqs (list of list of int): List of ground truth gesture IDs for each sample.

    Returns:
        float: The Levenshtein score (Total Distance / Total Ground Truth Length).
    """
    total_distance = 0
    total_length = 0

    for pred, target in zip(predicted_seqs, target_seqs):
        # Calculate Levenshtein distance for this sequence pair
        dist = nltk.edit_distance(pred, target)
        total_distance += dist
        total_length += len(target)

    if total_length == 0:
        return 0.0

    return total_distance / total_length


# ==========================================
# Custom Loss Functions
# ==========================================
class LogSpaceSmoothingLoss(nn.Module):
    """
    Truncated MSE on log-probabilities to enforce temporal smoothness.

    Formula: Mean( Clamp( (LogP_t - LogP_{t-1})^2, 0, threshold^2 ) ) * lambda
    """

    def __init__(self, lambda_weight=None, threshold=None):
        super(LogSpaceSmoothingLoss, self).__init__()
        self.lambda_weight = (
            lambda_weight if lambda_weight is not None else TrainConfig.SMOOTHING_LAMBDA
        )
        self.threshold = (
            threshold if threshold is not None else TrainConfig.SMOOTHING_THRESHOLD
        )

    def forward(self, log_probs):
        """
        Args:
            log_probs (torch.Tensor): Tensor of shape (Batch, Time, Classes).
                                      Should be output of LogSoftmax.
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculate temporal difference: LogP_t - LogP_{t-1}
        # shape: (Batch, Time-1, Classes)
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Squared error
        sq_diff = diff**2

        # Truncate (Clamp) the error
        # If threshold is provided, clamp the squared error to threshold^2
        if self.threshold is not None:
            max_val = self.threshold**2
            sq_diff = torch.clamp(sq_diff, max=max_val)

        # Mean over all dimensions
        loss = torch.mean(sq_diff)

        return self.lambda_weight * loss


# ==========================================
# Encoding / Decoding / Post-Processing
# ==========================================
def rle_encode(frames, min_duration=5, background_class=0):
    """
    Converts frame-wise class predictions into a list of gesture IDs using Run-Length Encoding.
    Applies a minimum duration filter and removes the background class.

    Args:
        frames (list or np.array): Sequence of class IDs.
        min_duration (int): Minimum number of frames for a segment to be considered valid.
        background_class (int): The class ID representing background/silence.

    Returns:
        list of int: Ordered list of recognized gesture IDs.
    """
    if len(frames) == 0:
        return []

    # 1. Identify segments (Run-Length Encoding)
    segments = []
    if len(frames) > 0:
        current_label = frames[0]
        current_len = 1

        for i in range(1, len(frames)):
            if frames[i] == current_label:
                current_len += 1
            else:
                segments.append((current_label, current_len))
                current_label = frames[i]
                current_len = 1
        segments.append((current_label, current_len))

    # 2. Filter segments and extract labels
    final_gestures = []
    for label, length in segments:
        # Skip background
        if label == background_class:
            continue

        # Apply duration filter
        if length >= min_duration:
            final_gestures.append(int(label))

    return final_gestures


def predictions_to_string(sample_id, gesture_list):
    """
    Formats a prediction into the submission CSV format.

    Args:
        sample_id (str): The sequence identifier (e.g., "Session00001").
        gesture_list (list of int): List of predicted gesture IDs.

    Returns:
        str: Formatted string "SessionID,Label1,Label2,..."
    """
    if not gesture_list:
        return f"{sample_id}"

    labels_str = ",".join(map(str, gesture_list))
    return f"{sample_id},{labels_str}"
