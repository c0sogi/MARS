import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the Config class implementation.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    Config.seed_everything(seed)


def get_device() -> torch.device:
    """
    Returns the computing device (CPU or GPU) configured in the Config.

    Returns:
        torch.device: The device object.
    """
    return Config.DEVICE


def calculate_rmse(y_true, y_pred) -> float:
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted pixel intensities.
    Handles both numpy arrays and torch tensors.

    Args:
        y_true: Ground truth values (numpy array or torch tensor).
        y_pred: Predicted values (numpy array or torch tensor).

    Returns:
        float: The RMSE value.
    """
    # Convert torch tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are flattened to treat them as a list of pixel intensities
    # This ensures shape mismatches (e.g. (N, 1) vs (N,)) don't cause broadcasting errors
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_true_flat, y_pred_flat)

    # Return Root Mean Squared Error
    return np.sqrt(mse)
