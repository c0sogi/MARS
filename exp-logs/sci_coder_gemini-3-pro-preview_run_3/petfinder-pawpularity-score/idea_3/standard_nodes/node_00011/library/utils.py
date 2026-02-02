import os
import sys
import random
import logging
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_file_path: str = None, name: str = "logger"):
    """
    Sets up a logger that outputs to both console and a file.

    Args:
        log_file_path (str): Path to the log file. If None, defaults to 'train.log' in Config.WORKING_DIR.
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if log_file_path is None:
        log_file_path = os.path.join(Config.WORKING_DIR, "train.log")

    # Ensure directory exists
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def compute_rmse(y_true, y_pred):
    """
    Computes Root Mean Squared Error (RMSE).

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Estimated target values.

    Returns:
        float: The RMSE value.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))
