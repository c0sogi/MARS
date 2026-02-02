import torch
import numpy as np
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility by delegating to the Config class.
    """
    Config.seed_everything(seed)


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


def LaplaceLogLikelihood(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric as defined in the task.

    Metric Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: Ground truth FVC values (Tensor or ndarray).
        y_pred: Predicted FVC values (Tensor or ndarray).
        sigma: Predicted confidence/std_dev (Tensor or ndarray).

    Returns:
        The mean metric value (scalar Tensor).
    """
    # Convert inputs to torch tensors if they are numpy arrays
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(sigma, np.ndarray):
        sigma = torch.from_numpy(sigma)

    # Ensure float precision
    y_true = y_true.float()
    y_pred = y_pred.float()
    sigma = sigma.float()

    # Move tensors to the same device (based on prediction tensor)
    device = y_pred.device
    y_true = y_true.to(device)
    sigma = sigma.to(device)

    # 1. Clip Confidence (Sigma)
    # The metric clips confidence at 70ml to reflect approximate measurement uncertainty
    sigma_clipped = torch.clamp(sigma, min=Config.MIN_CONFIDENCE)

    # 2. Calculate Clipped Absolute Error (Delta)
    # The error is thresholded at 1000ml to avoid large errors adversely penalizing results
    abs_diff = torch.abs(y_true - y_pred)
    delta = torch.clamp(abs_diff, max=Config.MAX_ERROR)

    # 3. Compute Metric
    # metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=device))

    term1 = (sqrt_2 * delta) / sigma_clipped
    term2 = torch.log(sqrt_2 * sigma_clipped)

    metric = -term1 - term2

    return torch.mean(metric)
