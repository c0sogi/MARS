import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for cuDNN backend.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the Log Loss metric (Binary Cross Entropy).

    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_pred (array-like): Predicted probabilities (between 0 and 1).

    Returns:
        float: The calculated log loss.
    """
    # sklearn's log_loss handles clipping (eps) internally to prevent log(0)
    return log_loss(y_true, y_pred)


def log_metric(name, value):
    """
    Logs a metric to the console with full precision.

    Args:
        name (str): The name of the metric.
        value (float): The value of the metric.
    """
    # Print without formatting to ensure full precision is displayed
    print(f"{name}: {value}")
