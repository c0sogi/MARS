import os
import sys
import random
import numpy as np
import torch
import logging
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(name="experiment", log_file=None, level=logging.INFO):
    """
    Sets up a logger that outputs to console and optionally to a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logs if setup is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class AverageMeter:
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


def log_metrics(logger, metrics, prefix=""):
    """
    Logs a dictionary of metrics with full precision.

    Args:
        logger (logging.Logger): The logger instance.
        metrics (dict): Dictionary of metric names and values.
        prefix (str): Optional prefix for the log message.
    """
    msg_parts = []
    if prefix:
        msg_parts.append(prefix)

    for k, v in metrics.items():
        if isinstance(v, (float, np.floating)):
            # Print full precision for floats as requested
            msg_parts.append(f"{k}: {v:.16f}")
        else:
            msg_parts.append(f"{k}: {v}")

    logger.info("  ".join(msg_parts))


def save_checkpoint(state, filename):
    """
    Saves the training state to a file.

    Args:
        state (dict): The state dictionary to save.
        filename (str): Path to the file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
