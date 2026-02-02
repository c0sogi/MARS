import os
import sys
import random
import logging
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed: int = 42):
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

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "main", filename: str = "main.log"):
    """
    Creates and configures a logger that writes to both console and a file.
    The log file is saved in the directory specified by Config.WORKING_DIR.

    Args:
        name (str): The name of the logger.
        filename (str): The name of the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if logger is already configured
    if logger.hasHandlers():
        return logger

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Ensure the working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    log_path = os.path.join(Config.WORKING_DIR, filename)

    # File Handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    return logger


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Computes the Log Loss metric for the competition.

    Args:
        y_true (np.ndarray): Ground truth labels or probabilities (N, 3).
        y_pred (np.ndarray): Predicted probabilities (N, 3).

    Returns:
        dict: A dictionary containing the 'log_loss' score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate Log Loss
    # eps='auto' is implicit in sklearn if we don't specify, but strict clipping helps stability
    # The competition metric specifies eps=auto behavior or standard log loss.
    score = log_loss(y_true, y_pred)

    return {"log_loss": score}
