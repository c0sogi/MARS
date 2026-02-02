import os
import random
import numpy as np
import torch
import logging
import sys
from sklearn.metrics import log_loss


def seed_everything(seed: int):
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
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "train", log_file: str = None):
    """
    Creates and configures a logger that writes to both console and a file.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to the log file. If provided, logs will be written to this file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File Handler
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def calculate_metric(y_true, y_pred):
    """
    Calculates the Log Loss metric.

    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_pred (array-like): Predicted probabilities for class 1.

    Returns:
        float: The calculated log loss.
    """
    # Epsilon is handled internally by sklearn's log_loss, but explicit clipping
    # is sometimes useful. We rely on sklearn implementation here as per standard.
    # labels parameter ensures it handles cases where a batch might only have one class.
    return log_loss(y_true, y_pred, labels=[0, 1])
