import os
import sys
import random
import logging
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(filename="train"):
    """
    Configures and returns a logger that writes to both a file and the console.

    Args:
        filename (str): The name of the log file.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Determine log path using Config
    log_dir = Config.WORKING_DIR
    os.makedirs(log_dir, exist_ok=True)

    if not filename.endswith(".log"):
        filename = f"{filename}.log"

    log_path = os.path.join(log_dir, filename)

    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    file_handler = logging.FileHandler(log_path, mode="w")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def get_score(y_true, y_pred):
    """
    Calculates the Area Under the Receiver Operating Curve (AUC).

    Args:
        y_true (array-like): Ground truth (binary) labels.
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The AUC score.
    """
    return roc_auc_score(y_true, y_pred)
