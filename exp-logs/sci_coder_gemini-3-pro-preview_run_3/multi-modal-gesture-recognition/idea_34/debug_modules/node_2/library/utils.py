import torch
import torch.nn as nn
import numpy as np
from library import config


def levenshtein_distance(preds, targets):
    """
    Computes the Levenshtein distance between two sequences of labels.

    Args:
        preds (list): List of predicted gesture IDs (integers).
        targets (list): List of ground truth gesture IDs (integers).

    Returns:
        int: The edit distance.
    """
    n = len(preds)
    m = len(targets)

    # Initialize matrix of zeros
    d = np.zeros((n + 1, m + 1), dtype=int)

    # Initialize first row and column
    for i in range(n + 1):
        d[i, 0] = i
    for j in range(m + 1):
        d[0, j] = j

    # Compute distances
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if preds[i - 1] == targets[j - 1] else 1
            d[i, j] = min(
                d[i - 1, j] + 1,  # Deletion
                d[i, j - 1] + 1,  # Insertion
                d[i - 1, j - 1] + cost,  # Substitution
            )

    return d[n, m]


def run_length_encoding(predictions, min_duration=config.MIN_GESTURE_DURATION):
    """
    Converts frame-wise predictions into a list of gesture IDs using Run-Length Encoding.
    Filters out background class and short segments.

    Args:
        predictions (np.ndarray or list): Array of frame-wise class IDs.
        min_duration (int): Minimum number of frames for a segment to be considered valid.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    if len(predictions) == 0:
        return []

    # Identify runs
    # Create a mask of where the value changes
    predictions = np.array(predictions)
    # Append a value different from the last to ensure the last run is captured if we use diff
    # Alternatively, manual iteration is robust and simple

    decoded_gestures = []
    current_label = predictions[0]
    current_len = 1

    for i in range(1, len(predictions)):
        label = predictions[i]
        if label == current_label:
            current_len += 1
        else:
            # Process the completed run
            if (
                current_label != config.BACKGROUND_CLASS_ID
                and current_len >= min_duration
            ):
                decoded_gestures.append(int(current_label))

            # Reset
            current_label = label
            current_len = 1

    # Process the final run
    if current_label != config.BACKGROUND_CLASS_ID and current_len >= min_duration:
        decoded_gestures.append(int(current_label))

    return decoded_gestures


class TruncatedMSELoss(nn.Module):
    """
    Computes Truncated Mean Squared Error on the temporal differences of log-probabilities.
    Used to enforce smoothness in predictions without over-penalizing sharp transitions.
    """

    def __init__(self, threshold=config.TRUNCATION_THRESHOLD):
        super(TruncatedMSELoss, self).__init__()
        self.threshold = threshold
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, input_log_probs):
        """
        Args:
            input_log_probs (torch.Tensor): Tensor of shape (Batch, Time, Classes) containing log-probabilities.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculate temporal difference: x[t] - x[t-1]
        # input shape: (B, T, C)
        if input_log_probs.size(1) < 2:
            return torch.tensor(0.0, device=input_log_probs.device, requires_grad=True)

        diff = input_log_probs[:, 1:, :] - input_log_probs[:, :-1, :]

        # Calculate squared error
        squared_diff = diff**2

        # Truncate the error
        # If diff^2 > threshold^2, clamp it.
        # This prevents large jumps (legitimate boundaries) from dominating the loss
        limit = self.threshold**2
        truncated_diff = torch.clamp(squared_diff, max=limit)

        return torch.mean(truncated_diff)


def compute_class_weights(device="cpu"):
    """
    Returns the class weights tensor based on configuration.

    Args:
        device (str or torch.device): Device to place the tensor on.

    Returns:
        torch.Tensor: Weights for CrossEntropyLoss.
    """
    weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32).to(device)
    return weights
