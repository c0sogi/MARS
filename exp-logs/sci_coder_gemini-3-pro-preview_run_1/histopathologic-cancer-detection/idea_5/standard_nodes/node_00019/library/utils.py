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
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enforce deterministic behavior for CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The current value to update.
            n (int): The weight/count of the value (usually batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true (np.array or list): Ground truth binary labels.
        y_pred (np.array or list): Predicted probabilities.

    Returns:
        float: The AUC score. Returns 0.5 if only one class is present in y_true.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle edge case where a batch/subset might only have one class
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)


def get_logger(name, log_file):
    """
    Creates and configures a logger that outputs to both file and stdout.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if the logger is retrieved multiple times
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Ensure the directory for the log file exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # File Handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Stream Handler (Console)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger
