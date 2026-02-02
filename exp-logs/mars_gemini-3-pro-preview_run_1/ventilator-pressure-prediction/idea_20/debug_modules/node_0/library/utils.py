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
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metric(y_pred, y_true, u_out):
    """
    Computes the Mean Absolute Error (MAE) for the inspiratory phase (u_out == 0).

    Args:
        y_pred (np.ndarray or torch.Tensor): Predicted pressure values.
        y_true (np.ndarray or torch.Tensor): Actual pressure values.
        u_out (np.ndarray or torch.Tensor): Control input u_out (0 for inspiratory, 1 for expiratory).

    Returns:
        float: The MAE calculated only over the inspiratory phase.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(u_out, torch.Tensor):
        u_out = u_out.detach().cpu().numpy()

    # Flatten arrays to ensure 1D alignment
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    u_out = u_out.flatten()

    # Identify the inspiratory phase (u_out == 0)
    # Using a boolean mask
    mask = u_out == 0

    # Handle edge case where there are no inspiratory steps (unlikely but safe)
    if np.sum(mask) == 0:
        return 0.0

    # Calculate MAE only for the masked elements
    mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))

    return mae
