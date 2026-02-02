import os
import sys
import random
import numpy as np
import torch
import logging
from scipy.stats import pearsonr


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
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
    Initializes and returns a logger that outputs to both console and a file.
    """
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Check if handlers already exist to avoid duplication
    if not logger.handlers:
        # Create handlers
        stream_handler = logging.StreamHandler(sys.stdout)

        # Ensure directory exists for the log file
        log_dir = os.path.dirname(filename)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(filename, mode="w")

        # Create formatters and add to handlers
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return logger


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_score(y_true, y_pred):
    """
    Calculates the Pearson correlation coefficient.

    Args:
        y_true: Array-like of ground truth scores.
        y_pred: Array-like of predicted scores.

    Returns:
        float: The Pearson correlation coefficient.
    """
    score = pearsonr(y_true, y_pred)[0]
    return float(score)


def get_device():
    """
    Returns the appropriate torch device (CUDA or CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
