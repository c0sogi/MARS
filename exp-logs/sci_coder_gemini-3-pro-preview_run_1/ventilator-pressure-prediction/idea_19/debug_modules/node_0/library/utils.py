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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metric(y_pred, y_true, u_out):
    """
    Computes the Mean Absolute Error (MAE) specifically on the inspiratory phase.
    The expiratory phase (where u_out == 1) is excluded from the metric.

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted pressure values.
        y_true (torch.Tensor or np.ndarray): Ground truth pressure values.
        u_out (torch.Tensor or np.ndarray): Control input 'u_out'.
                                            0 = inspiratory (scored),
                                            1 = expiratory (ignored).

    Returns:
        float: The calculated MAE for the inspiratory phase.
    """
    # Convert to torch tensors if inputs are numpy arrays for consistent processing
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(u_out, np.ndarray):
        u_out = torch.from_numpy(u_out)

    # Ensure inputs are on the same device
    if y_pred.device != y_true.device:
        y_true = y_true.to(y_pred.device)
    if u_out.device != y_pred.device:
        u_out = u_out.to(y_pred.device)

    # Create a boolean mask for the inspiratory phase (u_out == 0)
    mask = u_out == 0

    # Apply mask to flatten and select only relevant time steps
    y_pred_masked = y_pred[mask]
    y_true_masked = y_true[mask]

    # Check if mask is empty to avoid division by zero (though unlikely in valid data)
    if y_true_masked.numel() == 0:
        return 0.0

    # Compute Mean Absolute Error
    mae = torch.abs(y_pred_masked - y_true_masked).mean()

    return mae.item()
