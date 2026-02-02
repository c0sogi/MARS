import os
import sys
import random
import logging
import numpy as np
import torch
from scipy.stats import pearsonr


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(filename=None):
    """
    Initializes and returns a logger that outputs to both console and a file.

    Args:
        filename (str, optional): Path to the log file. If None, only logs to console.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    # Clean up existing handlers to avoid duplicates if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if filename:
        # Ensure directory exists
        log_dir = os.path.dirname(filename)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(filename, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def compute_score(y_true, y_pred):
    """
    Computes the Pearson correlation coefficient between true and predicted scores.

    Args:
        y_true (np.array or list): Ground truth scores.
        y_pred (np.array or list): Predicted scores.

    Returns:
        float: Pearson correlation coefficient.
    """
    # Ensure inputs are flattened numpy arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # pearsonr returns (statistic, p-value), we only need the statistic
    score, _ = pearsonr(y_true, y_pred)
    return score
