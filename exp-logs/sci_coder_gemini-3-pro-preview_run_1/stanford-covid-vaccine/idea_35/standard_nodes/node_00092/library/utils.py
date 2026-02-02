import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def mcrmse_loss(preds, targets):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) using PyTorch tensors.

    Args:
        preds (torch.Tensor): Predicted values. Shape (Batch, Seq_Len, Channels) or (N, Channels).
        targets (torch.Tensor): Ground truth values. Shape matches preds.

    Returns:
        torch.Tensor: Scalar MCRMSE value.
    """
    # Calculate MSE for each column (channel)
    # We reduce over all dimensions except the last one (channels)
    # If input is (Batch, Seq, Channels), dim=(0, 1)
    # If input is (N, Channels), dim=0
    reduction_dims = tuple(range(preds.ndim - 1))

    mse = torch.mean((preds - targets) ** 2, dim=reduction_dims)
    rmse = torch.sqrt(mse)

    # Average the RMSE across columns
    return torch.mean(rmse)


def metric_mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) using NumPy arrays.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: MCRMSE value.
    """
    # Calculate MSE for each column
    # We assume the last dimension is the channel dimension
    # Flatten all other dimensions to treat them as samples
    if y_true.ndim > 2:
        y_true = y_true.reshape(-1, y_true.shape[-1])
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    mse = np.mean((y_true - y_pred) ** 2, axis=0)
    rmse = np.sqrt(mse)
    return np.mean(rmse)


def masked_mse_loss(preds, targets, scored_len=Config.PRED_LENGTH):
    """
    Calculates the Mean Squared Error (MSE) only on the scored sequence positions.
    This aligns with the strategy to strictly use MSE (L2) loss on the first 68 bases.

    Args:
        preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, Channels).
        targets (torch.Tensor): Targets of shape (Batch, Seq_Len, Channels).
        scored_len (int): Number of positions to score from the start of the sequence.

    Returns:
        torch.Tensor: Scalar MSE loss.
    """
    # Slice to keep only the scored positions
    preds_scored = preds[:, :scored_len, :]
    targets_scored = targets[:, :scored_len, :]

    return F.mse_loss(preds_scored, targets_scored)
