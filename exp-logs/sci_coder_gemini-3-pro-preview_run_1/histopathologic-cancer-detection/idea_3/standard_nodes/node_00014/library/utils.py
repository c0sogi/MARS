import os
import random
import logging
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name, log_file=None):
    """
    Creates and returns a logger that writes to console and optionally to a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.
                                  If None, defaults to 'train.log' in Config.WORKING_DIR.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if log_file is None:
        log_file = os.path.join(Config.WORKING_DIR, "train.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File Handler
    # Ensure directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def compute_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve.

    Args:
        y_true (array-like or torch.Tensor): Ground truth binary labels.
        y_pred (array-like or torch.Tensor): Predicted probabilities.

    Returns:
        float: The AUC score.
    """
    # Detach tensors if necessary and convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Check if inputs are valid (e.g., not empty)
    if len(y_true) == 0:
        return 0.0

    try:
        score = roc_auc_score(y_true, y_pred)
    except ValueError:
        # Handle cases where only one class is present in y_true
        score = 0.0

    return score
