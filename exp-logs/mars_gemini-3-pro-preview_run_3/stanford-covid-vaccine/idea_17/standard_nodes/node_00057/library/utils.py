import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_loss(y_pred, y_true):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    Formula: Mean_over_columns( Sqrt( Mean_over_samples( (y - y_hat)^2 ) ) )

    Args:
        y_pred (torch.Tensor): Predicted values. Shape (Batch, ..., Num_Targets).
        y_true (torch.Tensor): Ground truth values. Shape (Batch, ..., Num_Targets).

    Returns:
        torch.Tensor: The scalar MCRMSE loss.
    """
    # Ensure inputs are float tensors
    y_pred = y_pred.float()
    y_true = y_true.float()

    # Calculate squared differences
    squared_diff = (y_pred - y_true) ** 2

    # Flatten all dimensions except the last one (targets)
    # This aggregates (Batch, Seq_Len) into a single sample dimension
    num_targets = y_pred.shape[-1]
    squared_diff_flat = squared_diff.view(-1, num_targets)

    # Calculate MSE for each target column
    mse = torch.mean(squared_diff_flat, dim=0)

    # Calculate RMSE for each target column
    rmse = torch.sqrt(mse)

    # Calculate Mean of RMSEs across all columns
    loss = torch.mean(rmse)

    return loss


def parse_structure_pairs(structure_sequence):
    """
    Parses a dot-bracket structure string into an adjacency index map.
    This map is used by the Structural Interaction Module to gather paired hidden states.

    Args:
        structure_sequence (str): A string representing RNA secondary structure
                                  (e.g., "...((...))...").

    Returns:
        numpy.ndarray: An integer array of length len(structure_sequence).
                       If position i is paired with j, arr[i] = j.
                       If position i is unpaired, arr[i] = -1.
    """
    seq_len = len(structure_sequence)
    # Initialize with -1 (indicating unpaired)
    pairs = np.full(seq_len, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure_sequence):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Record the pair in both directions
                pairs[i] = j
                pairs[j] = i
            else:
                # Handle potential malformed strings (unbalanced closing)
                # In this competition context, we leave it as -1
                pass

    return pairs
