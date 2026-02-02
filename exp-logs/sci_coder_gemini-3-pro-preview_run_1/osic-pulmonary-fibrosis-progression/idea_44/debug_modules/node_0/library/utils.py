import os
import random
import math
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
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
    Used for tracking loss and metrics during training.
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

    Metric formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: Ground truth FVC values (torch.Tensor or np.ndarray)
        y_pred: Predicted FVC values (torch.Tensor or np.ndarray)
        sigma: Predicted Confidence (sigma) values (torch.Tensor or np.ndarray)

    Returns:
        float: The average metric score across the batch.
    """
    # Initialize config to access metric constants
    cfg = Config()
    max_error = cfg.MAX_ERROR
    conf_clip = cfg.CONFIDENCE_CLIP

    # Convert inputs to torch tensors if they are numpy arrays
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, dtype=torch.float32)

    # Move to CPU to ensure consistent calculation and avoid device mismatch
    y_true = y_true.detach().cpu()
    y_pred = y_pred.detach().cpu()
    sigma = sigma.detach().cpu()

    # Clip confidence (sigma)
    # sigma_clipped = max(sigma, 70)
    sigma_clipped = torch.clamp(sigma, min=conf_clip)

    # Calculate absolute error and clip it (Delta)
    # Delta = min(|true - pred|, 1000)
    abs_diff = torch.abs(y_true - y_pred)
    delta = torch.clamp(abs_diff, max=max_error)

    # Calculate Metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    sqrt_2 = math.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - torch.log(sqrt_2 * sigma_clipped)

    # Return the mean metric
    return metric.mean().item()
