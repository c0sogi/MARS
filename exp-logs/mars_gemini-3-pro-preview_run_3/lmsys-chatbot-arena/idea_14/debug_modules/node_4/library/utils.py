import os
import sys
import time
import math
import random
import logging
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """
    Returns the appropriate torch device (CUDA or CPU).

    Returns:
        torch.device: The device to use for tensor computations.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_logger(log_file=None):
    """
    Initializes a logger that outputs to both console and a file.

    Args:
        log_file (str, optional): Path to the log file. If None, only logs to console.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File Handler
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class AverageMeter:
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


def as_minutes(s):
    """
    Converts seconds to a string format 'Mm Ss'.

    Args:
        s (float): Time in seconds.

    Returns:
        str: Formatted time string.
    """
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def time_since(since, percent):
    """
    Calculates elapsed time and estimated remaining time based on progress.

    Args:
        since (float): Timestamp when the process started.
        percent (float): Current progress percentage (0.0 to 1.0).

    Returns:
        str: String containing elapsed time and remaining time.
    """
    now = time.time()
    s = now - since
    if percent > 0:
        es = s / percent
        rs = es - s
        return "%s (remain %s)" % (as_minutes(s), as_minutes(rs))
    else:
        return "%s (remain ?)" % (as_minutes(s))
