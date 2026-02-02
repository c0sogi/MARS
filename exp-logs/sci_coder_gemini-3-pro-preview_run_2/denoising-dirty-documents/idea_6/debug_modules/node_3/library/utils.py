import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the Config class implementation to ensure consistency.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    Config.set_seed(seed)


def get_device() -> torch.device:
    """
    Retrieves the computation device (CPU or GPU) as defined in the configuration.

    Returns:
        torch.device: The device object.
    """
    return torch.device(Config.DEVICE)


def calculate_rmse(y_true, y_pred) -> float:
    """
    Calculates the Root Mean Squared Error (RMSE) between the ground truth and predictions.
    This function handles both NumPy arrays and PyTorch tensors.

    Args:
        y_true (np.ndarray or torch.Tensor): The ground truth pixel intensities.
        y_pred (np.ndarray or torch.Tensor): The predicted pixel intensities.

    Returns:
        float: The calculated RMSE value.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are flattened for simple element-wise comparison if they are not already
    # However, mean() works over all axes, so explicit flattening is not strictly required
    # for the math, but ensures safety against shape mismatches if they were to occur.
    # We assume shapes are compatible (e.g. both (N, C, H, W) or both flattened).

    # Use float64 for high precision calculation
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    return float(rmse)
