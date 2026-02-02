import os
import sys
import random
import logging
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility
    across random, numpy, and torch libraries.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "exp", log_file: str = None):
    """
    Configures and returns a logger instance that writes to stdout and an optional file.
    Prevents duplicate handlers if the logger is retrieved multiple times.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if get_logger is called repeatedly
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Stream Handler (Console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File Handler
        if log_file is not None:
            # Ensure directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

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


def get_score(y_true, y_pred):
    """
    Calculates the average ROC AUC score across all labels.
    Handles cases where a label might be missing from a batch by skipping it.

    Args:
        y_true: numpy array of shape (n_samples, n_classes) containing ground truth.
        y_pred: numpy array of shape (n_samples, n_classes) containing predicted probabilities.

    Returns:
        float: The average ROC AUC score.
    """
    scores = []
    num_classes = y_true.shape[1]

    for i in range(num_classes):
        # roc_auc_score requires at least one positive and one negative sample
        # We check unique values or catch the ValueError
        try:
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
            scores.append(score)
        except ValueError:
            # This happens if a class has only one unique value in y_true (e.g. all 0s)
            # In a full validation set this shouldn't happen often, but in batches it might.
            pass

    if len(scores) == 0:
        return 0.0

    return np.mean(scores)
