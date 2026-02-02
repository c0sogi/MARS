import os
import sys
import random
import numpy as np
import torch
import logging
from library.config import CFG


def seed_everything(seed: int = 42):
    """
    Seeds all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_logger(log_file=None):
    """
    Initializes a logger that outputs to both a file and stdout.
    """
    if log_file is None:
        os.makedirs(CFG.output_dir, exist_ok=True)
        log_file = os.path.join(CFG.output_dir, "train.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if function is called repeatedly
    if not logger.handlers:
        # Create handlers
        stream_handler = logging.StreamHandler(sys.stdout)
        file_handler = logging.FileHandler(log_file)

        # Create formatters and add it to handlers
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return logger


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
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
    Calculates categorization accuracy.

    Args:
        y_true: numpy array of shape (N,) containing true class indices.
        y_pred: numpy array of shape (N, C) containing logits or probabilities.

    Returns:
        float: Accuracy score.
    """
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return (y_pred == y_true).mean()


def print_metrics(logger, epoch, metrics):
    """
    Logs metrics with full precision as required.
    """
    msg = f"Epoch {epoch}: "
    # Using str() or repr() ensures no rounding occurs compared to f-string formatting like :.4f
    metric_strs = [f"{k}={v}" for k, v in metrics.items()]
    msg += " ".join(metric_strs)
    logger.info(msg)
