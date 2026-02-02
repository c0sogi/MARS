import logging
import os
import sys
import numpy as np
import torch
from library.config import Config, seed_everything


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and metrics during training.
    """

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def get_logger(log_file):
    """
    Sets up a logger that writes to console and a file.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(Config.PROJECT_NAME)
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplication if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_file)

    # Create formatters and add it to handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def kl_divergence_score(y_true, y_pred):
    """
    Calculates the Kullback-Leibler Divergence between predicted probabilities and observed targets.

    Metric = (1/N) * sum_i sum_j y_true_ij * (log(y_true_ij) - log(y_pred_ij))

    Args:
        y_true (np.array or torch.Tensor): Ground truth probabilities. Shape (N, 6).
        y_pred (np.array or torch.Tensor): Predicted probabilities. Shape (N, 6).

    Returns:
        float: The average KL divergence score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Clip predictions to avoid log(0)
    # Using a small epsilon
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Compute KL Divergence
    # D(P || Q) = sum(P(x) * log(P(x) / Q(x)))
    #           = sum(P(x) * log(P(x)) - P(x) * log(Q(x)))

    # Term 1: P(x) * log(P(x))
    # Handle 0 * log(0) = 0
    # We create a mask for non-zero true values
    term1 = np.zeros_like(y_true)
    mask = y_true > 0
    term1[mask] = y_true[mask] * np.log(y_true[mask])

    # Term 2: P(x) * log(Q(x))
    term2 = y_true * np.log(y_pred)

    # Sum over classes (axis 1)
    kl_per_sample = np.sum(term1 - term2, axis=1)

    # Mean over samples
    return np.mean(kl_per_sample)
