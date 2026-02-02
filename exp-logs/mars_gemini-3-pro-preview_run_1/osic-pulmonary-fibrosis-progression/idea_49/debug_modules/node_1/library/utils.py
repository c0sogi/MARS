import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def score_function(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric as defined in the task.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array or list): Ground truth FVC values.
        y_pred (np.array or list): Predicted FVC values.
        sigma (np.array or list): Predicted Confidence (standard deviation).

    Returns:
        float: The average metric score across all samples.
    """
    # Convert inputs to numpy arrays for vectorized operations
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    sigma = np.array(sigma, dtype=np.float32)

    # Retrieve constants from Config
    min_confidence = Config.MIN_CONFIDENCE
    max_error = Config.MAX_ERROR

    # Apply clipping to confidence (sigma)
    sigma_clipped = np.maximum(sigma, min_confidence)

    # Calculate absolute error (delta) and apply threshold
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, max_error)

    # Calculate metric
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    # Return the mean metric across the batch/dataset
    return np.mean(metric)
