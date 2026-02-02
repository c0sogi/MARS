import sys
import os
import random
import numpy as np
import torch
import logging
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred):
    """
    Calculates the Pearson correlation coefficient between true and predicted scores.

    Args:
        y_true (array-like): Ground truth scores.
        y_pred (array-like): Predicted scores.

    Returns:
        float: The Pearson correlation coefficient.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Check for empty arrays
    if len(y_true) < 2:
        return 0.0

    # Check for constant values to avoid division by zero (NaN result)
    # Pearson correlation is undefined if the standard deviation of either variable is 0
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0

    # Use numpy's corrcoef which returns the correlation matrix
    return np.corrcoef(y_true, y_pred)[0, 1]


def get_logger(name="main", log_dir=None):
    """
    Creates and configures a logger that outputs to both a file and the console.

    Args:
        name (str): The name of the logger.
        log_dir (str, optional): The directory to store the log file.
                                 Defaults to Config.working_dir if None.

    Returns:
        logging.Logger: The configured logger instance.
    """
    if log_dir is None:
        log_dir = Config.working_dir

    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logging if get_logger is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    log_file_path = os.path.join(log_dir, f"{name}.log")
    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_formatter = logging.Formatter("%(message)s")
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    return logger
