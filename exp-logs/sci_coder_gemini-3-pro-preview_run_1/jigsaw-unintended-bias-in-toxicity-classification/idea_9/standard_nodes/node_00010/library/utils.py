import os
import math
import time
import random
import logging
import sys
import numpy as np
import torch
from library.config import CFG


def seed_everything(seed: int = CFG.seed):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(filename: str = None):
    """
    Initializes and returns a logger that outputs to both console and a file.

    Args:
        filename (str, optional): Path to the log file. If None, defaults to 'train.log'
                                  inside CFG.output_dir.
    """
    if filename is None:
        filename = os.path.join(CFG.output_dir, "train.log")

    # Create directory if it doesn't exist
    log_dir = os.path.dirname(filename)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(CFG.project_name)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if logger is already configured
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(filename, mode="a")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Stream Handler (Console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_formatter = logging.Formatter("%(message)s")
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

    return logger


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Used for tracking loss and metrics during training.
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
    Converts seconds to 'm s' format.
    """
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    """
    Calculates elapsed time and estimated remaining time based on current progress.

    Args:
        since (float): Start time (time.time()).
        percent (float): Current progress percentage (0.0 to 1.0).
    """
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))
