import os
import sys
import logging
import shutil
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config, seed_everything


def get_logger(name: str = "main", log_file: str = None) -> logging.Logger:
    """
    Creates and configures a logger that writes to both stdout and a file.

    Args:
        name: Name of the logger.
        log_file: Path to the log file. If None, defaults to 'train.log' in Config.working_dir.

    Returns:
        Configured logging.Logger instance.
    """
    if log_file is None:
        log_file = os.path.join(Config.working_dir, "train.log")

    # Ensure directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def calculate_roc_auc(y_true, y_pred) -> float:
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels (numpy array or torch tensor).
        y_pred: Predicted probabilities (numpy array or torch tensor).

    Returns:
        float: ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten arrays to ensure 1D
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    # Check if we have both classes to calculate AUC
    if len(np.unique(y_true)) < 2:
        # If only one class is present, AUC is undefined.
        # Returning 0.5 is a neutral fallback, or we could return None.
        # For validation loops, this usually implies a bad batch or empty set,
        # but here we assume it's used for full epoch validation.
        return 0.5

    return roc_auc_score(y_true, y_pred)


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


def save_checkpoint(state, is_best, filepath=None, best_filepath=None):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filepath (str): Path to save the current checkpoint.
        best_filepath (str): Path to save the best model copy.
    """
    if filepath is None:
        filepath = os.path.join(Config.checkpoints_dir, "checkpoint.pth")

    torch.save(state, filepath)

    if is_best:
        if best_filepath is None:
            best_filepath = os.path.join(Config.checkpoints_dir, "best_model.pth")
        shutil.copyfile(filepath, best_filepath)
