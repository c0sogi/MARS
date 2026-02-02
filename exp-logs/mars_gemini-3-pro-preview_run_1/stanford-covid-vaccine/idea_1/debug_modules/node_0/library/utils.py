import torch
import numpy as np
import os
import random
from library.config import Config


def seed_everything(seed: int = None):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int, optional): The seed value to use. If None, uses Config.SEED.
    """
    if seed is None:
        seed = Config.SEED

    # Use the static method provided in Config to avoid re-implementation
    Config.set_seed(seed)


def get_device() -> torch.device:
    """
    Determines and returns the PyTorch device to be used.

    Returns:
        torch.device: The device (cuda or cpu) defined in Config.
    """
    return torch.device(Config.DEVICE)


def mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    Formula: Mean over columns of (SQRT(Mean over samples of (Error^2)))

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values of shape (N, C).
        y_pred (np.ndarray or torch.Tensor): Predicted values of shape (N, C).

    Returns:
        float: The calculated MCRMSE score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate MSE for each column (mean over samples/rows)
    # axis=0 represents the sample dimension
    mse_per_col = np.mean((y_true - y_pred) ** 2, axis=0)

    # Calculate RMSE for each column
    rmse_per_col = np.sqrt(mse_per_col)

    # Calculate the mean of the RMSEs (MCRMSE)
    score = np.mean(rmse_per_col)

    return float(score)
