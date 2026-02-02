import os
import random
import numpy as np
import torch
import logging
import sys
from scipy.stats import pearsonr


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

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


def get_score(y_true, y_pred):
    """
    Calculates the Pearson correlation coefficient.

    Args:
        y_true (array-like): Ground truth scores.
        y_pred (array-like): Predicted scores.

    Returns:
        float: The Pearson correlation coefficient.
    """
    score = pearsonr(y_true, y_pred)[0]
    return score


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


def get_logger(filename):
    """
    Initializes and configures a logger.

    Args:
        filename (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

    logger = getLogger(__name__)
    logger.setLevel(INFO)

    # Clear existing handlers to avoid duplicate logs if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    handler1 = StreamHandler(sys.stdout)
    handler1.setFormatter(Formatter("%(message)s"))

    handler2 = FileHandler(filename=filename)
    handler2.setFormatter(Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(handler1)
    logger.addHandler(handler2)

    return logger
