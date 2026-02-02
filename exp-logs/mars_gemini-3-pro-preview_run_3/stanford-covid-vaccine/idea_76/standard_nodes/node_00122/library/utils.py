import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

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


def mcrmse(y_true: torch.Tensor, y_pred: torch.Tensor, scored_indices: list = None):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    Formula: Mean over columns of (RMSE per column).

    Args:
        y_true (torch.Tensor): Ground truth tensor of shape (N, ..., C).
        y_pred (torch.Tensor): Predicted tensor of shape (N, ..., C).
        scored_indices (list, optional): List of integer indices corresponding to the
                                         columns to be included in the metric.
                                         If None, all columns are used.

    Returns:
        torch.Tensor: Scalar tensor containing the MCRMSE value.
    """
    # Ensure inputs are float tensors
    y_true = y_true.float()
    y_pred = y_pred.float()

    # If specific columns are requested, slice the last dimension
    if scored_indices is not None:
        y_true = y_true[..., scored_indices]
        y_pred = y_pred[..., scored_indices]

    # Calculate Squared Error: (y - y_hat)^2
    squared_error = (y_true - y_pred) ** 2

    # Calculate Mean Squared Error per column
    # We flatten all dimensions except the last one (channels/targets)
    # shape: (Total_Samples, C)
    flat_squared_error = squared_error.view(-1, squared_error.shape[-1])

    # Mean over the samples dimension (dim 0)
    mse_per_column = torch.mean(flat_squared_error, dim=0)

    # Calculate RMSE per column
    rmse_per_column = torch.sqrt(mse_per_column)

    # Calculate Mean of RMSEs (MCRMSE)
    mcrmse_val = torch.mean(rmse_per_column)

    return mcrmse_val


def parse_structure_to_pairs(structure: str):
    """
    Parses a dot-bracket structure string into adjacency indices and a binary mask
    for the Topology-Disentangled Interaction Module.

    Args:
        structure (str): Dot-bracket string (e.g., '...(((...)))...').

    Returns:
        tuple:
            - pair_indices (np.ndarray): Shape (L,). Index of the paired base.
              If unpaired, points to self (i).
            - pair_mask (np.ndarray): Shape (L,). 1.0 if paired, 0.0 if unpaired.
    """
    seq_len = len(structure)

    # Initialize indices pointing to self (default for unpaired)
    pair_indices = np.arange(seq_len)

    # Initialize mask (0 for unpaired)
    pair_mask = np.zeros(seq_len, dtype=np.float32)

    # Stack to keep track of opening brackets
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Register pair (undirected graph logic)
                pair_indices[i] = j
                pair_indices[j] = i

                # Mark as paired
                pair_mask[i] = 1.0
                pair_mask[j] = 1.0
            else:
                # Should not happen in valid dot-bracket notation provided in this dataset
                pass

    return pair_indices, pair_mask
