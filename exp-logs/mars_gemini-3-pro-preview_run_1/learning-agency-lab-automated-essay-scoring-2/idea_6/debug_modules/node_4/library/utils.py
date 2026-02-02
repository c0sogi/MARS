import os
import sys
import random
import logging
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from library.config import Config


def seed_everything(seed=Config.seed):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(filename=os.path.join(Config.working_dir, "train.log")):
    """
    Initializes and returns a logger that outputs to both console and a file.
    """
    logger = logging.getLogger("essay_scoring_logger")
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if the logger is already configured
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console Handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_formatter = logging.Formatter("%(message)s")
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

    return logger


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa (QWK) score.

    Args:
        y_true: Array-like of true scores (integers).
        y_pred: Array-like of predicted scores (floats or integers).

    Returns:
        float: The QWK score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # If predictions are continuous (from regression), clip to [1, 6] and round
    if y_pred.dtype.kind in "fc":  # float or complex
        y_pred = np.round(np.clip(y_pred, 1, 6)).astype(int)

    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


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
