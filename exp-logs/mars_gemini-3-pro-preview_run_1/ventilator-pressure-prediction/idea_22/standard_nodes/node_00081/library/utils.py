import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Determines the available hardware device for computation.

    Returns:
        torch.device: The device to be used (cuda or cpu).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def compute_metric(
    y_pred: torch.Tensor, y_true: torch.Tensor, u_out: torch.Tensor
) -> float:
    """
    Computes the Mean Absolute Error (MAE) for the inspiratory phase.

    The metric is defined as the MAE between predicted and actual pressures
    ONLY during the inspiratory phase (where u_out == 0).

    Args:
        y_pred (torch.Tensor): Predicted pressures.
        y_true (torch.Tensor): Ground truth pressures.
        u_out (torch.Tensor): Control input indicating expiratory phase (1) or inspiratory phase (0).

    Returns:
        float: The calculated MAE score.
    """
    # Ensure inputs are on the same device and detached from graph if necessary
    if y_pred.requires_grad:
        y_pred = y_pred.detach()

    # Create mask for inspiratory phase (u_out == 0)
    # u_out might be float or int, so we compare strictly or effectively
    mask = u_out == 0

    # Filter predictions and targets
    y_pred_insp = y_pred[mask]
    y_true_insp = y_true[mask]

    # Compute Absolute Error
    absolute_error = torch.abs(y_pred_insp - y_true_insp)

    # Return Mean
    if len(absolute_error) == 0:
        return 0.0

    return absolute_error.mean().item()
