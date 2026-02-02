import os
import sys
import random
import logging
import numpy as np
import torch
from sklearn.metrics import log_loss


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name: str = "project_logger", level=logging.INFO):
    """
    Sets up a simple logger that writes to stdout.

    Args:
        name (str): The name of the logger.
        level (int): The logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers if function is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def compute_metric(y_true, y_pred):
    """
    Computes the Log Loss metric with eps=auto (handled by sklearn default).

    Args:
        y_true (array-like): Ground truth labels (indices or one-hot).
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The log loss value.
    """
    # sklearn log_loss uses eps=1e-15 by default, which satisfies requirements
    return log_loss(y_true, y_pred)
