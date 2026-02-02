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

    # Ensure deterministic behavior for CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mae(y_pred, y_true, u_out):
    """
    Computes the Mean Absolute Error (MAE) specifically for the inspiratory phase.
    The expiratory phase (u_out=1) is excluded from the metric.

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted pressure values.
        y_true (torch.Tensor or np.ndarray): Actual pressure values.
        u_out (torch.Tensor or np.ndarray): Control input indicating expiratory valve status
                                            (0 for inspiratory, 1 for expiratory).

    Returns:
        float: The MAE for the inspiratory phase.
    """
    # Convert inputs to PyTorch tensors if they are NumPy arrays
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(u_out, np.ndarray):
        u_out = torch.from_numpy(u_out)

    # Ensure inputs are on the same device (CPU for metric calculation is usually sufficient/safer)
    y_pred = y_pred.detach().cpu()
    y_true = y_true.detach().cpu()
    u_out = u_out.detach().cpu()

    # Identify inspiratory phase indices: u_out == 0
    # Using < 0.5 to handle potential float representations of the binary flag
    inspiratory_mask = u_out < 0.5

    # Filter predictions and truth values using the mask
    y_pred_insp = y_pred[inspiratory_mask]
    y_true_insp = y_true[inspiratory_mask]

    # Check if there are any inspiratory samples to avoid division by zero
    if y_true_insp.numel() == 0:
        return 0.0

    # Calculate Mean Absolute Error
    mae = torch.abs(y_pred_insp - y_true_insp).mean()

    return mae.item()
