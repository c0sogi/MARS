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
    Computes the Mean Absolute Error (MAE) for the inspiratory phase.
    The metric is only calculated for time steps where u_out == 0.

    Args:
        y_pred (np.ndarray or torch.Tensor): Predicted pressures.
        y_true (np.ndarray or torch.Tensor): Ground truth pressures.
        u_out (np.ndarray or torch.Tensor): Control input for exploratory valve.
                                            0 represents inspiratory phase.

    Returns:
        float: The MAE calculated only where u_out == 0.
    """
    # Convert torch Tensors to numpy arrays if necessary
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(u_out, torch.Tensor):
        u_out = u_out.detach().cpu().numpy()

    # Flatten arrays to ensure 1D alignment regardless of input shape (e.g., [B, Seq] vs [B*Seq])
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    u_out = u_out.flatten()

    # Identify the inspiratory phase (u_out == 0)
    # Using a boolean mask for filtering
    mask = u_out == 0

    # Safety check: if no inspiratory phase exists in the batch (unlikely but possible)
    if np.sum(mask) == 0:
        return 0.0

    # Calculate Mean Absolute Error on the masked data
    # metric = mean( |y_true - y_pred| ) for u_out == 0
    mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))

    return mae
