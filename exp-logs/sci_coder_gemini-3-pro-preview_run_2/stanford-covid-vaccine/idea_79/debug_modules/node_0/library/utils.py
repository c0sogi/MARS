import os
import ast
import random
import numpy as np
import torch
import pandas as pd
from library.config import SCORED_INDICES


def set_seed(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_dot_bracket(structure):
    """
    Parses a dot-bracket structure string into a partner index map.
    This map is used to inject explicit partner identity into the model.

    Args:
        structure (str): A string representing RNA secondary structure
                         (e.g., "((..))").

    Returns:
        np.ndarray: An integer array of shape (len(structure),).
                    arr[i] is the index of the base paired with i,
                    or -1 if base i is unpaired.
    """
    n = len(structure)
    partner_map = np.full(n, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partner_map[i] = j
                partner_map[j] = i
            else:
                # Handle unbalanced closing brackets if necessary
                # For this task, we assume structures are largely valid
                pass

    return partner_map


def parse_list_column(x):
    """
    Parses a stringified list from the CSV metadata into a numpy array.
    Handles potential malformed strings or NaNs gracefully.

    Args:
        x (str or list): The value to parse.

    Returns:
        np.ndarray: A float32 numpy array. Returns an empty array if parsing fails.
    """
    try:
        if isinstance(x, str):
            # Evaluate the string literal (e.g., "[0.1, 0.2]")
            val = ast.literal_eval(x)
            return np.array(val, dtype=np.float32)
        elif isinstance(x, (list, tuple, np.ndarray)):
            return np.array(x, dtype=np.float32)
        else:
            return np.array([], dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)


def mcrmse(y_true, y_pred, scored_indices=None):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This function computes the global RMSE for each column and then averages them.
    It expects full arrays (concatenated across batches) to ensure the metric
    is calculated globally, not as an average of batch means.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
            Shape can be (N, C) or (N, SeqLen, C).
        y_pred (np.ndarray or torch.Tensor): Predicted values.
            Shape must match y_true.
        scored_indices (list, optional): List of column indices to include in the calculation.
            If None, all columns are used.

    Returns:
        float: The MCRMSE score.
    """
    # Convert Torch tensors to Numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten spatial dimensions if present: (N, SeqLen, C) -> (N*SeqLen, C)
    if y_true.ndim == 3:
        y_true = y_true.reshape(-1, y_true.shape[-1])
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    # Filter for specific scored columns if indices are provided
    if scored_indices is not None:
        # Ensure indices are valid for the input array dimensions
        valid_indices = [i for i in scored_indices if i < y_true.shape[1]]
        if valid_indices:
            y_true = y_true[:, valid_indices]
            y_pred = y_pred[:, valid_indices]

    # Calculate Mean Squared Error (MSE) for each column
    # axis=0 aggregates over all samples (global calculation)
    mse_per_column = np.mean((y_true - y_pred) ** 2, axis=0)

    # Calculate Root Mean Squared Error (RMSE) for each column
    rmse_per_column = np.sqrt(mse_per_column)

    # Calculate the mean of the columnwise RMSEs
    mcrmse_score = np.mean(rmse_per_column)

    return float(mcrmse_score)
