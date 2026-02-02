import numpy as np
import torch
from library.config import set_seed


def seed_everything(seed: int):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    This function wraps the set_seed function provided in the library configuration.

    Args:
        seed (int): The seed value to use.
    """
    set_seed(seed)


def calculate_rmse(y_true, y_pred):
    """
    Computes the Root Mean Squared Error (RMSE) between predicted and target values.
    This function handles both NumPy arrays and PyTorch tensors.

    Args:
        y_true (np.ndarray or torch.Tensor): The ground truth values.
        y_pred (np.ndarray or torch.Tensor): The predicted values.

    Returns:
        float: The calculated RMSE value.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays (handles lists, etc.)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate MSE
    mse = np.mean((y_true - y_pred) ** 2)

    # Calculate RMSE
    rmse = np.sqrt(mse)

    return rmse
