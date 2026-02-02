import torch
import numpy as np
from library.config import set_seed


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the set_seed function from library.config.

    Args:
        seed (int): The seed value to use.
    """
    set_seed(seed)


def mcrmse_metric(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated as the average of the RMSEs for each target column.

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth values.
            Expected shape: (Batch, Seq_Len, Channels) or (N, Channels).
        y_pred (torch.Tensor or np.ndarray): Predicted values.
            Must have the same shape as y_true.

    Returns:
        float: The calculated MCRMSE score.
    """
    # Convert numpy arrays to torch tensors if necessary
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)

    # Ensure inputs are float tensors
    y_true = y_true.float()
    y_pred = y_pred.float()

    # Check that shapes match
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Calculate Squared Error
    squared_error = (y_true - y_pred) ** 2

    # Calculate MSE per column (channel).
    # We average over all dimensions except the last one (which represents the columns/targets).
    # For shape (Batch, Seq_Len, 5), we average over dim 0 and 1.
    dims_to_reduce = tuple(range(y_true.dim() - 1))

    mse = torch.mean(squared_error, dim=dims_to_reduce)

    # Calculate RMSE per column
    rmse = torch.sqrt(mse)

    # Calculate Mean of RMSEs (MCRMSE) across the columns
    mcrmse = torch.mean(rmse)

    return mcrmse.item()
