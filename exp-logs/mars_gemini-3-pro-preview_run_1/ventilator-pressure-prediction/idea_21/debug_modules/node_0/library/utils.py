import os
import random
import numpy as np
import torch
from library.config import SEED


def seed_everything(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mae(y_pred, y_true, u_out):
    """
    Calculates the Mean Absolute Error (MAE) between predicted and actual pressures,
    considering only the inspiratory phase (where u_out == 0).

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted pressure values.
        y_true (torch.Tensor or np.ndarray): Ground truth pressure values.
        u_out (torch.Tensor or np.ndarray): Control input 'u_out' indicating phase.
                                            0 = inspiratory, 1 = expiratory.

    Returns:
        float: The MAE for the inspiratory phase.
    """
    # Convert tensors to numpy arrays if necessary, detaching from graph
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(u_out, torch.Tensor):
        u_out = u_out.detach().cpu().numpy()

    # Flatten arrays to ensure element-wise operations work correctly regardless of shape
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    u_out = u_out.flatten()

    # Create mask for inspiratory phase (u_out == 0)
    # Using < 0.5 to handle potential float representations of binary data safely
    mask = u_out < 0.5

    # Check if there are any inspiratory phase samples to avoid division by zero
    if np.sum(mask) == 0:
        return 0.0

    # Filter predictions and truth
    y_pred_insp = y_pred[mask]
    y_true_insp = y_true[mask]

    # Calculate MAE
    mae = np.mean(np.abs(y_true_insp - y_pred_insp))

    return float(mae)
