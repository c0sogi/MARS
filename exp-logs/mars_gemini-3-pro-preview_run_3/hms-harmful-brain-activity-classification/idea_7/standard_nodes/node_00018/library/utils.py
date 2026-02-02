import os
import random
import numpy as np
import torch
import logging
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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


def kl_divergence_score(y_true, y_pred, epsilon=1e-15):
    """
    Computes the Kullback-Leibler Divergence between true and predicted probabilities.
    Formula: sum(y_true * log(y_true / y_pred))

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth probabilities. Shape (N, C).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities. Shape (N, C).
        epsilon (float): Small constant to avoid numerical instability (log(0)).

    Returns:
        float: The average KL divergence score.
    """
    # Convert torch tensors to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Clip predicted probabilities to avoid log(0) or division by zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate KL Divergence: sum(P * (log(P) - log(Q)))
    # We handle P=0 by using a mask, as lim(x->0) x*log(x) = 0

    log_y_pred = np.log(y_pred)
    mask = y_true > 0

    kl_terms = np.zeros_like(y_true)

    # Only compute terms where y_true > 0 to avoid log(0) on ground truth
    if mask.any():
        kl_terms[mask] = y_true[mask] * (np.log(y_true[mask]) - log_y_pred[mask])

    # Sum over classes (axis=1)
    raw_kl = np.sum(kl_terms, axis=1)

    # Mean over samples
    return np.mean(raw_kl)


def get_logger(log_file):
    """
    Creates and returns a logger that writes to both a file and the console.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if function is called repeatedly
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)

        # Stream handler (Console)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

    return logger
