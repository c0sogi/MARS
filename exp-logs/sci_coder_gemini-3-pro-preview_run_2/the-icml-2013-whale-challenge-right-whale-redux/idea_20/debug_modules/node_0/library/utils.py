import os
import sys
import random
import logging
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
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
    Useful for tracking loss and metrics during training.
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
    Calculates the Area Under the ROC Curve.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: ROC AUC score. Returns 0.5 if only one class is present in y_true.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ROC AUC is undefined if there is only one class in the target
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)


def setup_logger(log_file: str):
    """
    Sets up a logger that writes to both a file and the console (stdout).

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("RightWhale")
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if the logger is already configured
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(file_handler)

    # Stream Handler (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    return logger


def save_checkpoint(state, is_best_auc, is_best_loss, checkpoint_dir, fold, model_name):
    """
    Saves the model checkpoint to the specified directory.

    Args:
        state (dict): The model state dictionary (model weights, optimizer, etc.).
        is_best_auc (bool): Whether this checkpoint has the best AUC so far.
        is_best_loss (bool): Whether this checkpoint has the best Loss so far.
        checkpoint_dir (str): Directory to save checkpoints.
        fold (int): Current fold number.
        model_name (str): Name of the model architecture.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    if is_best_auc:
        filename = f"{model_name}_fold_{fold}_best_auc.pth"
        path = os.path.join(checkpoint_dir, filename)
        torch.save(state, path)

    if is_best_loss:
        filename = f"{model_name}_fold_{fold}_best_loss.pth"
        path = os.path.join(checkpoint_dir, filename)
        torch.save(state, path)
