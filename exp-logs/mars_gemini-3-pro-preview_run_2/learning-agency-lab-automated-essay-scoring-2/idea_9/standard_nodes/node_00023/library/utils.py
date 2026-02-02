import os
import sys
import random
import numpy as np
import torch
import logging
from sklearn.metrics import cohen_kappa_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
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
    Calculates the Quadratic Weighted Kappa (QWK) score.
    Handles conversion from continuous regression predictions to integer classes.

    Args:
        y_true: Array-like of true scores (integers 1-6).
        y_pred: Array-like of predicted scores (can be floats).

    Returns:
        float: The QWK score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Round and clip predictions to match the valid score range [1, 6]
    # This ensures regression outputs are compatible with QWK
    y_pred = np.round(y_pred).clip(1, 6).astype(int)
    y_true = y_true.astype(int)

    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def get_logger(name="train"):
    """
    Configures and returns a logger instance.
    Logs are written to stdout and a file in the working directory defined in Config.

    Args:
        name (str): The name of the logger. Defaults to "train".

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if get_logger is called multiple times
    if logger.hasHandlers():
        return logger

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Stream Handler (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File Handler
    # Ensure the working directory exists (Config.setup handles this, but safe check)
    os.makedirs(Config.working_dir, exist_ok=True)
    log_file = os.path.join(Config.working_dir, f"{name}.log")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
