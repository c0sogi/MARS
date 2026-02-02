import os
import sys
import random
import logging
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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


def compute_qwk(y_true, y_pred):
    """
    Computes the Quadratic Weighted Kappa (QWK) score.

    This function handles the conversion of continuous regression predictions
    to the required integer scale [1, 6] by rounding and clipping.

    Args:
        y_true (array-like): True scores (integers).
        y_pred (array-like): Predicted scores (can be continuous floats).

    Returns:
        float: The Quadratic Weighted Kappa score.
    """
    # Convert Torch tensors to Numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Round predictions to the nearest integer and clip to the valid score range [1, 6]
    y_pred = np.round(y_pred).clip(1, 6).astype(int)
    y_true = y_true.astype(int)

    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def get_logger(name, log_file=None):
    """
    Initializes and configures a logger for tracking training progress.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to the log file. If provided, logs will also be saved to this file.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding duplicate handlers if the logger is retrieved multiple times
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Stream Handler (Output to Console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File Handler (Output to File)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics like loss during training loops.
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
