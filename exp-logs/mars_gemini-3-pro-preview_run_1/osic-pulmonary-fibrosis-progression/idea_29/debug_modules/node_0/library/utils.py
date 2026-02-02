import os
import sys
import logging
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across various libraries.
    Wraps the Config.set_seed method to avoid re-implementation.

    Args:
        seed (int): The random seed to set.
    """
    Config.set_seed(seed)


def calculate_metric(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array or torch.Tensor): Ground truth FVC values.
        y_pred (np.array or torch.Tensor): Predicted FVC values.
        sigma (np.array or torch.Tensor): Predicted Confidence (standard deviation).

    Returns:
        float: The mean metric score over the input batch.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Ensure inputs are float for calculation
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    sigma = sigma.astype(float)

    # 1. Clip confidence values
    # The confidence values are clipped at 70 ml to reflect approximate measurement uncertainty
    sigma_clipped = np.maximum(sigma, 70)

    # 2. Calculate absolute error and threshold it
    # The error is thresholded at 1000 ml to avoid large errors adversely penalizing results
    abs_error = np.abs(y_true - y_pred)
    delta = np.minimum(abs_error, 1000)

    # 3. Compute Metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    sqrt_2 = np.sqrt(2)
    term1 = (sqrt_2 * delta) / sigma_clipped
    term2 = np.log(sqrt_2 * sigma_clipped)

    metric = -term1 - term2

    return np.mean(metric)


def get_logger(name="training"):
    """
    Configures and returns a logger instance.
    Logs are output to stdout and a file in the working directory.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)

    # If logger already has handlers, assume it's configured and return it
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    # Ensure the output directory exists
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    log_file_path = os.path.join(Config.OUTPUT_DIR, "train.log")

    fh = logging.FileHandler(log_file_path, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


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
