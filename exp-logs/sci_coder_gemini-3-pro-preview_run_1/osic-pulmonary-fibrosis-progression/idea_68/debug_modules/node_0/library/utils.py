import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Seeds all random number generators for reproducibility.
    Wraps the configuration's seeding method.
    """
    Config.seed_everything(seed)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking metrics and losses during training.
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


def laplace_log_likelihood(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric.

    The metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: Ground truth FVC values (Tensor or numpy array).
        y_pred: Predicted FVC values (Tensor or numpy array).
        sigma: Predicted Confidence (std dev) values (Tensor or numpy array).

    Returns:
        The mean metric score (scalar tensor).
    """
    # Convert inputs to torch tensors if they are numpy arrays
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, dtype=torch.float32)

    # Ensure all tensors are on the same device
    device = y_pred.device
    y_true = y_true.to(device)
    sigma = sigma.to(device)

    # Apply clipping constraints defined in the metric
    sigma_clipped = torch.clamp(sigma, min=Config.MIN_CONFIDENCE_CLIP)
    delta = torch.abs(y_true - y_pred)
    delta_clipped = torch.clamp(delta, max=Config.MAX_ERROR_CLIP)

    # Calculate metric
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=device))
    metric = -(sqrt_2 * delta_clipped) / sigma_clipped - torch.log(
        sqrt_2 * sigma_clipped
    )

    return torch.mean(metric)


def loss_fn(y_true, y_pred, sigma):
    """
    Loss function that mirrors the evaluation metric for optimization.

    Calculates the Negative Log Likelihood with specific clipping:
    Loss = -Metric

    Crucially, this includes the error clipping at 1000ml to filter outliers
    during training, as specified in the task description.

    Args:
        y_true: Ground truth FVC values.
        y_pred: Predicted FVC values.
        sigma: Predicted Confidence values.

    Returns:
        The mean loss (scalar tensor).
    """
    # Ensure consistency with device
    device = y_pred.device

    # Apply clipping constraints
    # Gradients for errors > 1000ml will be zeroed out by clamp, providing robustness
    sigma_clipped = torch.clamp(sigma, min=Config.MIN_CONFIDENCE_CLIP)
    delta = torch.abs(y_true - y_pred)
    delta_clipped = torch.clamp(delta, max=Config.MAX_ERROR_CLIP)

    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=device))

    # Calculate Loss (Negative Metric)
    loss = (sqrt_2 * delta_clipped) / sigma_clipped + torch.log(sqrt_2 * sigma_clipped)

    return torch.mean(loss)
