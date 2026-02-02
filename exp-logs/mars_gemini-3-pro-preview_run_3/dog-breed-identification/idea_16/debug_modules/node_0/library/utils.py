import os
import sys
import random
import logging
import warnings
import numpy as np
import torch
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

    # Ensure deterministic behavior for cuDNN to guarantee reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, log_file: str):
    """
    Configures and returns a logger that writes to both console and a file.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplication if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler (stdout)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def calculate_score(y_true, y_pred):
    """
    Computes the Multi Class Log Loss.

    Args:
        y_true: Ground truth labels. Can be 1D array of indices or 2D one-hot encoded array.
        y_pred: Predicted probabilities. 2D array of shape (n_samples, n_classes).

    Returns:
        float: The calculated log loss.
    """
    # Use sklearn's log_loss implementation
    # We suppress warnings that might arise from epsilon clipping or normalization
    # to keep the output clean as per requirements.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = log_loss(y_true, y_pred)

    return score
