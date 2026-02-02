import os
import sys
import random
import logging
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=Config.SEED):
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
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


def get_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Target scores (probability estimates of the positive class).

    Returns:
        float: The ROC AUC score. Returns 0.5 if only one class is present in y_true.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check if both classes are present to avoid ValueError from sklearn
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)


def setup_logger(name="WSIL_Logger", log_file=None):
    """
    Sets up a logger to print messages to stdout and optionally to a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to a file where logs should be saved.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if logger is already configured
    if not logger.handlers:
        # Console Handler
        c_handler = logging.StreamHandler(sys.stdout)
        c_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)

        # File Handler (optional)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            f_handler = logging.FileHandler(log_file)
            f_handler.setFormatter(c_format)
            logger.addHandler(f_handler)

    return logger
