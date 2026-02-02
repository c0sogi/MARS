import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated as the average of the RMSE values for each scored column.

    Args:
        y_true (np.ndarray): Ground truth values. Shape can be (N, C) or (N, L, C).
        y_pred (np.ndarray): Predicted values. Shape must match y_true.

    Returns:
        float: The MCRMSE score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Identify the number of target columns (the last dimension)
    num_columns = y_true.shape[-1]

    # Flatten all dimensions except the last one to handle (Batch, Seq, Channels)
    # or (Samples, Channels) uniformly.
    y_true_flat = y_true.reshape(-1, num_columns)
    y_pred_flat = y_pred.reshape(-1, num_columns)

    # Calculate MSE for each column
    mse_per_col = np.mean((y_true_flat - y_pred_flat) ** 2, axis=0)

    # Calculate RMSE for each column
    rmse_per_col = np.sqrt(mse_per_col)

    # Return the mean of the column-wise RMSEs
    return np.mean(rmse_per_col)
