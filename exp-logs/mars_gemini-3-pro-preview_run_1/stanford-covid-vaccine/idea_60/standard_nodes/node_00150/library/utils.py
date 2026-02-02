import torch
import numpy as np
from library.config import set_seed, mcrmse_loss


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Default is 42.
    """
    set_seed(seed)


def calculate_mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This function computes the RMSE for each of the 3 scored columns
    (reactivity, deg_Mg_pH10, deg_Mg_50C) separately across all samples
    and positions, and then returns the mean of these RMSEs.

    It handles both NumPy arrays and PyTorch tensors, and ensures inputs
    are flattened correctly to compute the metric over the entire dataset.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
            Expected shape: (N, 3) or (N, L, 3).
        y_pred (np.ndarray or torch.Tensor): Predicted values.
            Expected shape: (N, 3) or (N, L, 3).

    Returns:
        float: The MCRMSE score.
    """
    # Convert numpy arrays to torch tensors if necessary
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)

    # Ensure data is on the CPU and float type for metric calculation
    y_true = y_true.cpu().float()
    y_pred = y_pred.cpu().float()

    # Flatten the batch and sequence dimensions to treat all positions as samples for each column
    # Shape becomes (Total_Positions, 3)
    # This is critical because the metric is defined as the average of the RMSEs of the columns.
    # If we don't flatten, we might average over batches first, which is incorrect.
    y_true = y_true.reshape(-1, 3)
    y_pred = y_pred.reshape(-1, 3)

    # Use the imported loss function which calculates mean(sqrt(mean((y-y_hat)^2, dim=0)))
    # dim=0 here reduces the flattened dimension (Total_Positions), resulting in RMSE per column.
    score = mcrmse_loss(y_true, y_pred)

    return score.item()
