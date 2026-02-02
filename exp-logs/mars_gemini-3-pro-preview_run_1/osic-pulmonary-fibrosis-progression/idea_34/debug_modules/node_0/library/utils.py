import os
import random
import math
import numpy as np
import torch


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking losses and metrics during training.
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


def seed_everything(seed=42):
    """
    Sets seeds for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def score(y_true, y_pred, sigma, max_error=1000, confidence_clip=70):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Metric formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth FVC.
        y_pred (torch.Tensor or np.ndarray): Predicted FVC.
        sigma (torch.Tensor or np.ndarray): Predicted Confidence (standard deviation).
        max_error (float): Threshold to clip the absolute error (default: 1000).
        confidence_clip (float): Minimum value for confidence (default: 70).

    Returns:
        float: The average metric score across the batch/dataset.
    """
    # Convert numpy arrays to torch tensors for unified processing
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(sigma, np.ndarray):
        sigma = torch.from_numpy(sigma)

    # Ensure inputs are detached and on CPU for calculation
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu()
    if torch.is_tensor(sigma):
        sigma = sigma.detach().cpu()

    # 1. Clip Confidence (Sigma)
    # sigma_clipped = max(sigma, 70)
    sigma_clipped = torch.maximum(sigma, torch.tensor(float(confidence_clip)))

    # 2. Calculate Absolute Error and Clip (Delta)
    # delta = min(|FVC_true - FVC_pred|, 1000)
    abs_error = torch.abs(y_true - y_pred)
    delta = torch.minimum(abs_error, torch.tensor(float(max_error)))

    # 3. Calculate Metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    sqrt_2 = math.sqrt(2)

    term1 = -(sqrt_2 * delta) / sigma_clipped
    term2 = -torch.log(sqrt_2 * sigma_clipped)

    metric = term1 + term2

    # Return the mean metric over the batch
    return torch.mean(metric).item()
