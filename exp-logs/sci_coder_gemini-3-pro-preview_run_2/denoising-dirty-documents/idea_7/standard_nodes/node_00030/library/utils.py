import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the PyTorch device based on availability and Config.

    Returns:
        torch.device: The device (CPU or CUDA).
    """
    return torch.device(Config.DEVICE)


def calculate_rmse(y_true, y_pred) -> float:
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.
    Supports both numpy arrays and torch tensors.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        float: The RMSE value.
    """
    # Convert torch tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten arrays to ensure global RMSE calculation across all pixels
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
    return np.sqrt(mse)
