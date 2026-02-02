import os
import sys
import random
import numpy as np
import torch
import logging
from sklearn.metrics import roc_auc_score


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Enforces deterministic behavior in CuDNN backends.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.

    # Enforce deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # print(f"Random seed set to {seed} and deterministic algorithms enabled.")


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (ROC AUC).

    Args:
        y_true (np.ndarray): Ground truth binary labels (N_samples, N_classes).
        y_pred (np.ndarray): Predicted probabilities (N_samples, N_classes).

    Returns:
        float: Macro-averaged ROC AUC score.
    """
    # Cite debug_lesson_5: Safeguard Global Metrics Against Degenerate Data Subsets
    # Manually iterate over classes to handle cases where a class has no positive samples
    # in the current batch/subset (which causes roc_auc_score to return NaN or raise error).
    class_aucs = []
    num_classes = y_true.shape[1]

    for i in range(num_classes):
        # Only calculate AUC if the class has both positive and negative samples
        if len(np.unique(y_true[:, i])) == 2:
            class_aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))

    if len(class_aucs) == 0:
        return 0.0

    return np.mean(class_aucs)


def get_logger(name, log_file=None):
    """
    Configures and returns a logger instance.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Stream Handler (Console)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


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
