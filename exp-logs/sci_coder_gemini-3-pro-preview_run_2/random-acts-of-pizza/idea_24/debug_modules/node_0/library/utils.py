import os
import sys
import random
import logging
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed: int = Config.RANDOM_SEED):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use. Defaults to Config.RANDOM_SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(name: str = "pshle", log_file: str = None, level=logging.INFO):
    """
    Configures and returns a logger instance.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file. If None, only logs to stdout.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times if function is called repeatedly
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Stream Handler (Console)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File Handler
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def compute_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (AUC-ROC).

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Target scores or probabilities.

    Returns:
        float: The AUC-ROC score.
    """
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError as e:
        # Handle cases where only one class is present in y_true
        logging.getLogger("pshle").warning(f"Error calculating AUC: {e}")
        return 0.5
