import os
import random
import numpy as np
import torch
import logging
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Enforces deterministic algorithms in CUDNN.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Macro-Averaged ROC AUC score for multi-label classification.

    Args:
        y_true (np.ndarray): Ground truth binary labels of shape (N_samples, N_classes).
        y_pred (np.ndarray): Predicted probabilities of shape (N_samples, N_classes).

    Returns:
        float: The macro-averaged ROC AUC score. Returns 0.5 if calculation fails
               (e.g., only one class present in a mini-batch).
    """
    try:
        # Calculate Macro-Averaged ROC AUC
        score = roc_auc_score(y_true, y_pred, average="macro")
        return score
    except ValueError:
        # Handle cases where y_true might not have both classes (0 and 1) for a specific label
        # This can happen during debugging with very small batch sizes.
        return 0.5


def get_logger(name="experiment"):
    """
    Configures and returns a logger instance for tracking experiment progress.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
