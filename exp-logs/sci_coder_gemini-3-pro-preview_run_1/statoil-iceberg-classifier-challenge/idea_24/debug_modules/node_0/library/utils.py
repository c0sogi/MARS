import os
import sys
import random
import logging
import numpy as np
import torch
from library import config


def seed_everything(seed=config.RANDOM_SEED):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The random seed to use. Defaults to config.RANDOM_SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name, log_file=None):
    """
    Configures and returns a logger instance that writes to both a file and stdout.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to the log file. If None, defaults to 'train.log' in WORKING_DIR.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if log_file is None:
        log_file = os.path.join(config.WORKING_DIR, "train.log")

    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs if get_logger is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

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


def get_device():
    """
    Returns the torch device based on availability and config.

    Returns:
        torch.device: The device to be used for computation.
    """
    return torch.device(config.DEVICE)
