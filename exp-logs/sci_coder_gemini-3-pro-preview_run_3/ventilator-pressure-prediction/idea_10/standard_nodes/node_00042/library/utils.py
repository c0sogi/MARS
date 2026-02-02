import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Ensure deterministic behavior for CuDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the PyTorch device (CUDA or CPU) based on configuration.

    Returns:
        torch.device: The device object.
    """
    return torch.device(Config.DEVICE)


def compute_mae(y_pred, y_true, u_out):
    """
    Calculates the Mean Absolute Error (MAE) specifically for the inspiratory phase.

    The metric is defined as the MAE between predicted and actual pressures
    only for time steps where u_out == 0 (inspiratory phase). The expiratory
    phase (u_out == 1) is excluded from the calculation.

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted pressures.
        y_true (torch.Tensor or np.ndarray): Ground truth pressures.
        u_out (torch.Tensor or np.ndarray): Control input indicating expiratory phase (1) or inspiratory (0).

    Returns:
        float: The calculated MAE score.
    """
    # Ensure inputs are tensors
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, device=get_device())
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, device=get_device())
    if not isinstance(u_out, torch.Tensor):
        u_out = torch.tensor(u_out, device=get_device())

    # Flatten inputs to ensure shape consistency (e.g., if (Batch, Seq, 1) vs (Batch, Seq))
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    u_out = u_out.view(-1)

    # Create mask for inspiratory phase (u_out == 0)
    # We use a threshold or exact equality depending on data type, but 0 is exact for this binary feature
    mask = u_out == 0

    # Check if there are any valid samples to avoid division by zero
    if mask.sum() == 0:
        return 0.0

    # Calculate Absolute Error
    absolute_error = torch.abs(y_pred - y_true)

    # Apply mask to select only inspiratory phase errors
    masked_error = absolute_error[mask]

    # Calculate Mean
    mae = masked_error.mean().item()

    return mae
