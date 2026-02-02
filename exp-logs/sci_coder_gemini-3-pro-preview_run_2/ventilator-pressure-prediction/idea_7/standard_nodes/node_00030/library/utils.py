import os
import random
import numpy as np
import torch
from library.config import SEED


def seed_everything(seed: int = SEED):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
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


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device (CUDA or CPU).

    Returns:
        torch.device: The device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def compute_metric(preds, targets, u_out):
    """
    Computes the Mean Absolute Error (MAE) for the inspiratory phase.
    The metric is only calculated where u_out == 0.

    Args:
        preds (torch.Tensor or np.ndarray): Predicted pressure values.
        targets (torch.Tensor or np.ndarray): Actual pressure values.
        u_out (torch.Tensor or np.ndarray): Control input indicating expiratory phase (1) or inspiratory (0).

    Returns:
        float: The MAE for the inspiratory phase.
    """
    # Convert tensors to numpy if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(u_out, torch.Tensor):
        u_out = u_out.detach().cpu().numpy()

    # Flatten arrays to ensure 1D alignment
    preds = preds.flatten()
    targets = targets.flatten()
    u_out = u_out.flatten()

    # Filter for inspiratory phase (u_out == 0)
    inspiratory_mask = u_out == 0

    if np.sum(inspiratory_mask) == 0:
        return 0.0

    # Calculate MAE
    errors = np.abs(preds[inspiratory_mask] - targets[inspiratory_mask])
    return np.mean(errors)


def log_metric(name: str, value: float):
    """
    Prints a metric with full precision without rounding or formatting.

    Args:
        name (str): The name of the metric.
        value (float): The value of the metric.
    """
    print(f"{name}: {value}")
