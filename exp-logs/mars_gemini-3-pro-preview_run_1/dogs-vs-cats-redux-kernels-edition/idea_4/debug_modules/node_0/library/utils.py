import os
import sys
import time
import math
import random
import logging
import numpy as np
import torch
from library.config import CFG


def seed_everything(seed: int = 42):
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
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(filename: str):
    """
    Initializes and configures a logger.

    Args:
        filename (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if get_logger is called repeatedly
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(filename, mode="a")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(file_handler)

        # Stream handler (stdout)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

    return logger


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
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


def asMinutes(s):
    """
    Converts seconds to minutes and seconds string.
    """
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    """
    Calculates elapsed time and estimated remaining time.

    Args:
        since (float): Start time.
        percent (float): Progress percentage (0.0 to 1.0).

    Returns:
        str: Formatted string "Elapsed (- Remaining)"
    """
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


def save_checkpoint(state, is_best, filepath):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint is the best so far.
        filepath (str): Path to save the checkpoint.
    """
    torch.save(state, filepath)
    # If we wanted to save a separate 'best' copy, we could do it here,
    # but the filepath usually indicates if it's best or fold specific.


def print_metrics(metrics_dict, logger=None):
    """
    Prints validation metrics with full precision.

    Args:
        metrics_dict (dict): Dictionary of metric names and values.
        logger (logging.Logger, optional): Logger to use. If None, prints to stdout.
    """
    msg_parts = []
    for k, v in metrics_dict.items():
        msg_parts.append(f"{k}: {v}")

    msg = " | ".join(msg_parts)

    if logger:
        logger.info(msg)
    else:
        print(msg)
