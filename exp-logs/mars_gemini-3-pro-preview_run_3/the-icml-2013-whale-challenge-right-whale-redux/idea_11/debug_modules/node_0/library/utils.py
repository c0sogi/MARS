import os
import sys
import random
import logging
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to set.
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


def get_logger(name: str = "rwd_logger"):
    """
    Creates and returns a logger that prints to stdout.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if the logger is retrieved multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        # Use a simple format to avoid clutter and comply with print requirements
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def calculate_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve (AUC).

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The AUC score. Returns 0.5 if only one class is present or an error occurs.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    try:
        # roc_auc_score requires both classes to be present in y_true
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.5
