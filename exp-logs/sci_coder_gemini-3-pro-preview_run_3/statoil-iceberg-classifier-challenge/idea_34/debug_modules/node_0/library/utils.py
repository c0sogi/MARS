import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
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
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the Log Loss metric. Handles both numpy arrays and torch tensors.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels (0 or 1).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities.

    Returns:
        float: The calculated log loss.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # sklearn's log_loss handles clipping internally (eps=1e-15 by default)
    return log_loss(y_true, y_pred)


def get_device():
    """
    Returns the appropriate torch device based on availability.

    Returns:
        torch.device: 'cuda' if available, otherwise 'cpu'.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
