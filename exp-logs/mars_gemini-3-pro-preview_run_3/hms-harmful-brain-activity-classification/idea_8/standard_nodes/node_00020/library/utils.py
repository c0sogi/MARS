import os
import sys
import random
import numpy as np
import torch
import logging


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure fully reproducible results.

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


def kl_divergence(y_true, y_pred):
    """
    Calculates the Kullback-Leibler (KL) divergence between the predicted probability
    distribution and the observed target distribution.

    Metric Formula: sum(P(x) * (log(P(x)) - log(Q(x))))

    Args:
        y_true (np.ndarray): Ground truth probabilities of shape (N, C).
        y_pred (np.ndarray): Predicted probabilities of shape (N, C).

    Returns:
        float: The average KL divergence across the batch.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Clip targets safely for the log operation to avoid log(0)
    # Note: y_true * log(y_true) approaches 0 as y_true -> 0
    y_true_safe = np.clip(y_true, epsilon, 1.0)

    # Calculate KL Divergence
    # Sum over classes (axis=1), then mean over samples
    kl_div = np.sum(y_true * (np.log(y_true_safe) - np.log(y_pred)), axis=1)

    return np.mean(kl_div)


def get_logger(log_file):
    """
    Configures and returns a logger that writes to both a file and the console (stdout).

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers to prevent duplication if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Ensure the directory for the log file exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training epochs.
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
