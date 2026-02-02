import os
import random
import numpy as np
import torch
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


def calculate_metric(y_true, y_pred, sigma):
    """
    Calculates the Modified Laplace Log Likelihood metric.

    Metric formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: Ground truth FVC values (torch.Tensor or np.array).
        y_pred: Predicted FVC values (torch.Tensor or np.array).
        sigma: Predicted Confidence/Std Dev values (torch.Tensor or np.array).

    Returns:
        float: The mean metric score for the batch.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Ensure sigma is absolute (though model output should be positive)
    sigma = np.abs(sigma)

    # 1. Clip the confidence (sigma)
    # The confidence values are clipped at 70 ml
    sigma_clipped = np.maximum(sigma, Config.Q_CLIP)

    # 2. Calculate the error (Delta)
    # The error is thresholded at 1000 ml
    absolute_diff = np.abs(y_true - y_pred)
    delta = np.minimum(absolute_diff, Config.MAX_ERR)

    # 3. Compute the metric
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    # Return the average score
    return np.mean(metric)
