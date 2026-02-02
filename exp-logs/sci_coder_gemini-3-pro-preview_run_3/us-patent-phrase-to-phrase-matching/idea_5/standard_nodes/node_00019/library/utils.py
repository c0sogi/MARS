import os
import sys
import random
import logging
import numpy as np
import torch
from scipy.stats import pearsonr
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(filename="train"):
    """
    Initializes and returns a logger that outputs to both a file and stdout.
    The log file is saved in Config.working_dir.
    """
    log_file = os.path.join(Config.working_dir, f"{filename}.log")

    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplication if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


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


def get_score(y_true, y_pred):
    """
    Computes the Pearson correlation coefficient between true and predicted values.
    """
    score = pearsonr(y_true, y_pred)[0]
    return score
