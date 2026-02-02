import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    MCRMSE is the mean of the RMSE values calculated for each target column separately.

    Args:
        y_true (torch.Tensor): Ground truth tensor. Shape (N, C) or (B, L, C).
        y_pred (torch.Tensor): Predicted tensor. Shape (N, C) or (B, L, C).

    Returns:
        torch.Tensor: Scalar MCRMSE loss.
    """
    # Ensure inputs are float
    y_true = y_true.float()
    y_pred = y_pred.float()

    # Calculate Squared Error
    squared_error = (y_true - y_pred) ** 2

    # Calculate MSE per column
    # If input is 3D (Batch, Seq, Channels), average over Batch (0) and Seq (1)
    # If input is 2D (N, Channels), average over N (0)
    if squared_error.dim() == 3:
        mse_per_col = torch.mean(squared_error, dim=(0, 1))
    else:
        mse_per_col = torch.mean(squared_error, dim=0)

    # Calculate RMSE per column
    rmse_per_col = torch.sqrt(mse_per_col)

    # Calculate Mean of RMSEs
    mcrmse = torch.mean(rmse_per_col)

    return mcrmse


def parse_structure(structure: str):
    """
    Parses a dot-bracket structure string into pairing indices and distance maps.

    Args:
        structure (str): A string representing RNA secondary structure in dot-bracket notation
                         (e.g., "((..))").

    Returns:
        tuple: A tuple containing two numpy arrays:
            - pair_index (np.ndarray): Array of shape (L,) where arr[i] is the index of the base
              paired with i, or -1 if unpaired.
            - distances (np.ndarray): Array of shape (L,) where arr[i] is the absolute distance
              |i - j| if i is paired with j, or 0 if unpaired.
    """
    n = len(structure)
    pair_index = np.full(n, -1, dtype=np.int32)
    distances = np.zeros(n, dtype=np.int32)

    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start = stack.pop()
                # Register the pair (start, i)
                pair_index[start] = i
                pair_index[i] = start

                # Calculate distance
                dist = abs(i - start)
                distances[start] = dist
                distances[i] = dist

    return pair_index, distances
