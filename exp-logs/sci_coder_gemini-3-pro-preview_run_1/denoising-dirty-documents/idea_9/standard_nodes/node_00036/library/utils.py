import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure deterministic behavior.

    Args:
        seed (int): The seed value to set. Defaults to the value in Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # PyTorch seeding
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # CuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between the cleaned pixel intensities
    and the actual grayscale pixel intensities.

    Handles both NumPy arrays and PyTorch tensors.

    Args:
        y_true: Ground truth values. Can be a numpy array or torch.Tensor.
        y_pred: Predicted values. Can be a numpy array or torch.Tensor.

    Returns:
        float: The calculated RMSE value.
    """
    # Convert torch tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Compute Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)

    # Compute Root Mean Squared Error
    rmse = np.sqrt(mse)

    return float(rmse)
