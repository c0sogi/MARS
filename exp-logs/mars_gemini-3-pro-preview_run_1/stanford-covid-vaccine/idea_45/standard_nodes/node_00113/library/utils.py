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

    # Ensure deterministic behavior in CuDNN backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def MCRMSE(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This function computes the RMSE for each of the 3 scored columns
    (reactivity, deg_Mg_pH10, deg_Mg_50C) separately, and then averages
    those RMSE values. This aligns with the competition metric.

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth values.
            Expected shape: (Batch, Seq_Len, 3) or (N, 3).
        y_pred (torch.Tensor or np.ndarray): Predicted values.
            Expected shape: Same as y_true.

    Returns:
        torch.Tensor: The scalar MCRMSE score.
    """
    # Convert numpy arrays to torch tensors if necessary
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    # Ensure tensors are on the same device
    if y_true.device != y_pred.device:
        y_true = y_true.to(y_pred.device)

    # Calculate squared errors: (y - y_hat)^2
    squared_diff = (y_true - y_pred) ** 2

    # Calculate MSE for each column (target) independently.
    # We reduce (mean) over all dimensions except the last one (the column dimension).
    # For input shape (Batch, Seq, 3), we mean over dims (0, 1).
    dims_to_reduce = list(range(y_true.ndim - 1))
    mse_per_column = torch.mean(squared_diff, dim=dims_to_reduce)

    # Calculate RMSE for each column
    rmse_per_column = torch.sqrt(mse_per_column)

    # Calculate the mean of the column-wise RMSEs
    mcrmse = torch.mean(rmse_per_column)

    return mcrmse
