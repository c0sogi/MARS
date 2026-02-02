import os
import sys
import random
import logging
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Seeds all random number generators to ensure reproducibility.

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


class AverageMeter:
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


def get_logger(filename):
    """
    Creates a logger that writes to a file and stdout.
    Ensures the directory for the log file exists.

    Args:
        filename (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_dir = os.path.dirname(filename)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if get_logger is called repeatedly
    if not logger.handlers:
        formatter = logging.Formatter("%(message)s")

        # File Handler
        fh = logging.FileHandler(filename, mode="w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Stream Handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger


def map_per_image(label, predictions):
    """
    Computes the average precision score for a single image.

    Args:
        label (str): The true label.
        predictions (list): A list of predicted elements (strings).

    Returns:
        float: The average precision score (1/rank if found, else 0).
    """
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0


def map_at_5(targets, predictions):
    """
    Computes the Mean Average Precision at 5 (MAP@5).

    Args:
        targets (list): A list of true labels (strings).
        predictions (list): A list of lists of predicted elements.

    Returns:
        float: The mean average precision at 5.
    """
    total = len(targets)
    if total == 0:
        return 0.0
    score = 0.0
    for t, p in zip(targets, predictions):
        score += map_per_image(t, p)
    return score / total
