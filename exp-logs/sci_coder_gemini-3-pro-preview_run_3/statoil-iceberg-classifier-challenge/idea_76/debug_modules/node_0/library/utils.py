import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_log_loss(y_true, y_pred, eps=1e-15):
    """
    Calculates the Log Loss metric.

    Args:
        y_true (array-like): Ground truth binary labels (0 or 1).
        y_pred (array-like): Predicted probabilities (between 0 and 1).
        eps (float): Epsilon value for clipping probabilities (handled by sklearn usually,
                     but good to be aware of).

    Returns:
        float: The calculated log loss.
    """
    # sklearn's log_loss handles clipping internally, but we pass raw predictions
    # y_pred should be probabilities of the positive class (Iceberg)
    return log_loss(y_true, y_pred, eps=eps)
