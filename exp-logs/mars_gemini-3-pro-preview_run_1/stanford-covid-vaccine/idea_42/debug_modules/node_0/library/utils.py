import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=None):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int, optional): The seed value to set. If None, uses Config.seed.
    """
    if seed is None:
        seed = Config.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def mcrmse_loss(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This function computes the RMSE for each of the 3 target columns separately
    and then returns the average of these RMSEs. It strictly evaluates on the
    first `Config.pred_len` (68) positions of the sequence, slicing the inputs
    if necessary.

    Args:
        y_true (torch.Tensor): Ground truth values. Expected shape (Batch, 68, 3)
                               or (Batch, Seq_Len, 3).
        y_pred (torch.Tensor): Predicted values. Expected shape (Batch, Seq_Len, 3).

    Returns:
        torch.Tensor: A scalar tensor containing the MCRMSE value.
    """
    # Ensure inputs are tensors
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)

    # Move y_true to the same device as y_pred
    if y_true.device != y_pred.device:
        y_true = y_true.to(y_pred.device)

    # Define the number of scored positions
    scored_len = Config.pred_len

    # Slice inputs to the scored length if they exceed it
    # We assume the shape is (Batch, Sequence_Length, Channels)
    if y_pred.shape[1] > scored_len:
        y_pred = y_pred[:, :scored_len, :]

    if y_true.shape[1] > scored_len:
        y_true = y_true[:, :scored_len, :]

    # Verify shapes match after slicing
    # Note: y_true should match y_pred in the first two dimensions (Batch, Scored_Len)
    # and the channel dimension (3)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch in mcrmse_loss: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Calculate MSE for each column (channel)
    # We average over the Batch (dim 0) and Sequence (dim 1) dimensions
    mse_per_column = torch.mean((y_true - y_pred) ** 2, dim=(0, 1))

    # Calculate RMSE for each column
    rmse_per_column = torch.sqrt(mse_per_column)

    # Calculate the mean of the column-wise RMSEs
    mcrmse = torch.mean(rmse_per_column)

    return mcrmse
