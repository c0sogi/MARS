import os
import sys
import random
import numpy as np
import torch
import logging
from scipy.stats import pearsonr
from library.config import CFG


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

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


def get_score(y_true, y_pred):
    """
    Calculates the Pearson correlation coefficient.

    Args:
        y_true (np.array or list): Ground truth scores.
        y_pred (np.array or list): Predicted scores.

    Returns:
        float: Pearson correlation coefficient.
    """
    score = pearsonr(y_true, y_pred)[0]
    return score


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
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


def setup_logger(out_file=None):
    """
    Sets up the logger to write to console and a file.

    Args:
        out_file (str): Path to the log file. If None, uses CFG.output_dir/train.log

    Returns:
        logging.Logger: The configured logger instance.
    """
    if out_file is None:
        os.makedirs(CFG.output_dir, exist_ok=True)
        out_file = os.path.join(CFG.output_dir, "train.log")
    else:
        # Ensure directory exists for the provided file path
        log_dir = os.path.dirname(out_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates if setup_logger is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    file_handler = logging.FileHandler(out_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
