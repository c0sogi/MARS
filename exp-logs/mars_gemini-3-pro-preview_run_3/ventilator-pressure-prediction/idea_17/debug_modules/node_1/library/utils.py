import torch
import numpy as np
import os
from library.config import set_seed


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the set_seed function from the configuration library.

    Args:
        seed (int): The seed value to use.
    """
    set_seed(seed)


def get_device():
    """
    Automatically selects the computing device.

    Returns:
        torch.device: 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_mae_loss(y_pred, y_true, u_out):
    """
    Calculates the Mean Absolute Error (MAE) between predicted and actual pressures,
    considering only the inspiratory phase of the breath.

    The metric is defined as the MAE where u_out == 0. The expiratory phase (u_out == 1)
    is ignored.

    Args:
        y_pred (torch.Tensor): The predicted pressure values.
        y_true (torch.Tensor): The ground truth pressure values.
        u_out (torch.Tensor): The control input 'u_out' indicating the phase
                              (0 for inspiratory, 1 for expiratory).

    Returns:
        torch.Tensor: The scalar masked MAE loss.
    """
    # Create a boolean mask where u_out is 0 (inspiratory phase).
    # Using < 0.5 ensures robustness against potential float representations of 0.
    mask = u_out < 0.5

    # Calculate the absolute error for all points
    error = torch.abs(y_pred - y_true)

    # Extract only the errors corresponding to the inspiratory phase
    masked_error = error[mask]

    # Return the mean of these errors.
    # If the mask is empty (no inspiratory phase), this would return NaN,
    # but physically every breath in this dataset has an inspiratory phase.
    return masked_error.mean()
