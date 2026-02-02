import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the PyTorch device (CPU or CUDA) based on availability and configuration.

    Returns:
        torch.device: The device object.
    """
    return torch.device(Config.DEVICE)


def compute_metric(
    preds: torch.Tensor, targets: torch.Tensor, u_out: torch.Tensor
) -> float:
    """
    Computes the Mean Absolute Error (MAE) specifically for the inspiratory phase.
    The expiratory phase (where u_out == 1) is excluded from the calculation.

    Args:
        preds (torch.Tensor): Predicted pressure values.
        targets (torch.Tensor): Actual pressure values.
        u_out (torch.Tensor): Control input for the exploratory valve (0 for inspiration, 1 for expiration).

    Returns:
        float: The MAE for the inspiratory phase.
    """
    # Ensure inputs are on the same device and detached from graph for metric calculation
    if preds.device != u_out.device:
        preds = preds.to(u_out.device)
    if targets.device != u_out.device:
        targets = targets.to(u_out.device)

    # The metric is only computed for the inspiratory phase (u_out == 0)
    # Create a boolean mask where True indicates the inspiratory phase
    mask = u_out == 0

    # Calculate absolute error
    abs_error = torch.abs(preds - targets)

    # Apply the mask to select only inspiratory phase errors
    masked_error = abs_error[mask]

    # If there are no inspiratory steps in this batch (unlikely but possible), return 0.0
    if masked_error.numel() == 0:
        return 0.0

    # Return the mean of the masked errors
    return masked_error.mean().item()
