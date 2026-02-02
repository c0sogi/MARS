import os
import random
import shutil
import logging
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
    """
    Seeding for reproducibility.
    Sets seeds for random, numpy, and torch (CPU & GPU).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking losses and metrics during training.
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


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels (numpy array).
        y_pred: Predicted probabilities (numpy array).

    Returns:
        float: ROC AUC score. Returns 0.5 if only one class is present.
    """
    # Roc AUC score requires both classes to be present in y_true
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)


def get_logger(name, log_file):
    """
    Creates a logger that writes to both a file and the console.

    Args:
        name: Name of the logger.
        log_file: Path to the log file.

    Returns:
        logging.Logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # prevent duplicate handlers
    if not logger.handlers:
        # Create formatters
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # File Handler
        # Ensure the directory for the log file exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        f_handler = logging.FileHandler(log_file)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

        # Stream Handler (Console)
        c_handler = logging.StreamHandler()
        c_handler.setFormatter(formatter)
        logger.addHandler(c_handler)

    return logger


def save_checkpoint(state, is_best, filepath):
    """
    Saves the model checkpoint.

    Args:
        state: Dict containing model state (weights, optimizer, etc.).
        is_best: Boolean indicating if this is the best model so far.
        filepath: Full path to save the checkpoint file.
    """
    # Ensure directory exists
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Save the checkpoint
    torch.save(state, filepath)

    # If it is the best model, copy it to a standard 'best_model.pth' name
    if is_best:
        best_path = os.path.join(directory, "best_model.pth")
        shutil.copyfile(filepath, best_path)
