import os
import sys
import random
import logging
import numpy as np
import torch
from scipy.stats import pearsonr


def seed_everything(seed=42):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        seed (int): The random seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(filename):
    """
    Creates and configures a logger that writes to both a file and standard output.

    Args:
        filename (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    # Remove existing handlers to prevent duplicate logging if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatters and handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # File Handler
    # Ensure directory exists for the log file
    log_dir = os.path.dirname(filename)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Used for tracking metrics like loss during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all internal statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the moving average.

        Args:
            val (float): The current value.
            n (int): The weight/count of the current value (default: 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_score(y_true, y_pred):
    """
    Calculates the Pearson correlation coefficient.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Estimated target values.

    Returns:
        float: The Pearson correlation coefficient.
    """
    # pearsonr returns (statistic, p-value), we only need the statistic
    score, _ = pearsonr(y_true, y_pred)
    return score
