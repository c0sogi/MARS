import os
import sys
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from library import config


def set_seed(seed=config.SEED):
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


def setup_logger(log_file_path):
    """
    Sets up a logger that writes to a file and the console.
    """
    # Create directory if it doesn't exist
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("KC-IRN")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger


def compute_levenshtein(hypothesis, reference):
    """
    Computes the Levenshtein distance between two sequences of gesture IDs.

    Args:
        hypothesis (list): List of predicted gesture IDs.
        reference (list): List of ground truth gesture IDs.

    Returns:
        int: The edit distance.
    """
    len_hyp = len(hypothesis)
    len_ref = len(reference)

    # Initialize DP matrix
    # dp[i][j] is distance between hyp[:i] and ref[:j]
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


class TruncatedMSELoss(nn.Module):
    """
    Truncated Mean Squared Error Loss applied to temporal differences of log-probabilities.
    Used to enforce temporal smoothness in predictions.

    L = mean( min( (log_prob[t] - log_prob[t-1])^2, threshold ) )
    """

    def __init__(self, threshold=1.0):
        super(TruncatedMSELoss, self).__init__()
        self.threshold = threshold

    def forward(self, input_log_probs):
        """
        Args:
            input_log_probs: Tensor of shape (Batch, Classes, Time) or (Batch, Time, Classes).
                             We assume (Batch, Classes, Time) to align with TCN outputs.
        """
        # Ensure we are working with (Batch, Classes, Time)
        if input_log_probs.dim() == 3 and input_log_probs.size(1) != config.NUM_CLASSES:
            # If shape is (Batch, Time, Classes), permute to (Batch, Classes, Time)
            input_log_probs = input_log_probs.permute(0, 2, 1)

        # Calculate difference between adjacent frames along the time dimension (dim=2)
        # diff[t] = input[t] - input[t-1]
        diff = input_log_probs[:, :, 1:] - input_log_probs[:, :, :-1]

        # Squared Error
        mse = diff**2

        # Truncate the error to prevent penalizing valid sharp transitions too heavily
        truncated_mse = torch.clamp(mse, max=self.threshold)

        return truncated_mse.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for dense classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_log_probs, target):
        """
        Args:
            input_log_probs: (Batch, Classes, Time)
            target: (Batch, Time)
        """
        if input_log_probs.dim() == 3:
            input_log_probs = input_log_probs.permute(0, 2, 1)  # (Batch, Time, Classes)

        # Flatten
        input_log_probs = input_log_probs.reshape(-1, input_log_probs.size(-1))
        target = target.reshape(-1)

        # Gather log_probs for target class
        log_pt = input_log_probs.gather(1, target.view(-1, 1)).view(-1)
        pt = log_pt.exp()

        # Focal term
        focal_term = (1 - pt).pow(self.gamma)

        # Alpha term
        if self.alpha is not None:
            if self.alpha.device != target.device:
                self.alpha = self.alpha.to(target.device)
            at = self.alpha.gather(0, target).view(-1)
            loss = -at * focal_term * log_pt
        else:
            loss = -focal_term * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def decode_predictions_to_sequence(
    frame_predictions, background_id=config.BACKGROUND_CLASS_ID, min_len=5
):
    """
    Decodes frame-wise class predictions into a sequence of gesture IDs.
    Applies Run-Length Encoding and filters out background and short segments.

    Args:
        frame_predictions (list or np.array): List of predicted class IDs for each frame.
        background_id (int): The ID representing the background/null class.
        min_len (int): Minimum duration (in frames) for a segment to be considered a valid gesture.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    if len(frame_predictions) == 0:
        return []

    # Run-Length Encoding
    segments = []
    if len(frame_predictions) > 0:
        current_label = frame_predictions[0]
        current_len = 1

        for label in frame_predictions[1:]:
            if label == current_label:
                current_len += 1
            else:
                segments.append((current_label, current_len))
                current_label = label
                current_len = 1
        segments.append((current_label, current_len))

    # Filter segments
    final_sequence = []
    for label, length in segments:
        if label != background_id:
            # For non-background gestures, apply length filter if needed
            # (Though often for competition metrics, we might want to be careful with filtering)
            if length >= min_len:
                final_sequence.append(int(label))

    return final_sequence
