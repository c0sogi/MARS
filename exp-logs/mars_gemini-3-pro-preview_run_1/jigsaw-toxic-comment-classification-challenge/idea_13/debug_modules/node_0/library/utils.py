import os
import sys
import random
import numpy as np
import torch
import logging
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
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
    Creates and configures a logger that writes to both a file and the console.

    Args:
        filename (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure the directory for the log file exists
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logging if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    # File Handler
    file_handler = logging.FileHandler(filename, mode="w")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    return logger


class AverageMeter(object):
    """
    Computes and stores the average and current value of a metric.
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
    Calculates the mean column-wise ROC AUC.

    Args:
        y_true (np.array or pd.DataFrame): Ground truth binary labels.
        y_pred (np.array or pd.DataFrame): Predicted probabilities.

    Returns:
        float: The mean column-wise ROC AUC score.
    """
    return roc_auc_score(y_true, y_pred, average="macro")
