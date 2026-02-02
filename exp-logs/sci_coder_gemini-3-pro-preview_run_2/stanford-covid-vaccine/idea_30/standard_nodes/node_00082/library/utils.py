import os
import random
import ast
import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed=42):
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


def parse_list_column(x):
    """
    Parses a string representation of a list (e.g., from a CSV) into a numpy array.

    Args:
        x (str or list): The input value, potentially a stringified list.

    Returns:
        np.ndarray: A float32 numpy array. Returns an empty array on failure.
    """
    try:
        if isinstance(x, (list, np.ndarray)):
            return np.array(x, dtype=np.float32)
        if isinstance(x, str):
            # ast.literal_eval safely evaluates a string containing a Python literal
            return np.array(ast.literal_eval(x), dtype=np.float32)
        return np.array([], dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)


def mcrmse_loss(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is computed by:
    1. Calculating the MSE for each column (target) across the batch/sequence.
    2. Taking the square root of each column's MSE to get RMSE.
    3. Averaging the RMSE values across all columns.

    Args:
        y_true (torch.Tensor): Ground truth tensor. Shape (Batch, Seq_Len, Channels) or (N, C).
        y_pred (torch.Tensor): Predicted tensor. Shape (Batch, Seq_Len, Channels) or (N, C).

    Returns:
        torch.Tensor: The scalar MCRMSE loss.
    """
    # Flatten the tensors to (N, C) to handle (Batch, Seq_Len, Channels) inputs
    if y_pred.dim() > 2:
        y_pred = y_pred.view(-1, y_pred.size(-1))
    if y_true.dim() > 2:
        y_true = y_true.view(-1, y_true.size(-1))

    # Calculate Mean Squared Error per column
    # dim=0 averages over the batch/sequence dimension
    mse = torch.mean((y_true - y_pred) ** 2, dim=0)

    # Calculate RMSE per column (adding epsilon for numerical stability)
    rmse = torch.sqrt(mse + 1e-8)

    # Calculate the mean of the RMSEs across columns
    loss = torch.mean(rmse)

    return loss


class MCRMSELoss(nn.Module):
    """
    PyTorch Module wrapper for the MCRMSE loss function.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return mcrmse_loss(y_true, y_pred)
