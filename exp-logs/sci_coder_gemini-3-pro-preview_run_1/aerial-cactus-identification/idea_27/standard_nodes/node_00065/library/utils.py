import os
import sys
import random
import logging
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str = Config.PROJECT_NAME, log_file: str = None):
    """
    Configures and returns a logger with console and optional file handlers.

    Args:
        name: Name of the logger.
        log_file: Path to the log file. If None, only console logging is enabled.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicate logs if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        # Ensure the directory for the log file exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels (1D array-like or Tensor).
        y_pred: Predicted probabilities for the positive class (1D array-like or Tensor).

    Returns:
        float: The ROC AUC score.
    """
    # Detach and move to CPU if inputs are PyTorch tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are flattened
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    try:
        score = roc_auc_score(y_true, y_pred)
    except ValueError:
        # This can happen if y_true has only one class (e.g., in a small batch)
        # Return 0.5 as a neutral score in such edge cases
        score = 0.5

    return score


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
