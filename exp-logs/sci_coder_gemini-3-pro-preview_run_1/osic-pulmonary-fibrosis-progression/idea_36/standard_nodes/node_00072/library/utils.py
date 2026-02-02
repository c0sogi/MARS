import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the modified Laplace Log Likelihood metric as a loss function.

    Metric Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Loss Formula (Minimization Objective):
        loss = -metric
        loss = (sqrt(2) * delta / sigma_clipped) + ln(sqrt(2) * sigma_clipped)

    Args:
        reduction (str): 'mean' or 'sum' or 'none'.
        for_training (bool): If True, delta (error) is NOT clipped at 1000ml to preserve gradients.
                             If False, delta is clipped as per the official metric definition.
    """

    def __init__(self, reduction="mean", for_training=True):
        super().__init__()
        self.reduction = reduction
        self.for_training = for_training
        self.sigma_clip = Config.SIGMA_CLIP
        self.max_error = Config.MAX_ERROR
        # Precompute sqrt(2)
        self.register_buffer("sqrt_2", torch.sqrt(torch.tensor(2.0)))

    def forward(self, fvc_pred, fvc_true, sigma_pred):
        """
        Args:
            fvc_pred (Tensor): Predicted FVC values.
            fvc_true (Tensor): Ground truth FVC values.
            sigma_pred (Tensor): Predicted confidence (sigma) values.
        """
        # Flatten tensors to ensure matching shapes
        fvc_pred = fvc_pred.view(-1)
        fvc_true = fvc_true.view(-1)
        sigma_pred = sigma_pred.view(-1)

        # Clip confidence (sigma)
        # Metric requires max(sigma, 70)
        sigma_clipped = torch.clamp(sigma_pred, min=self.sigma_clip)

        # Calculate absolute error
        delta = torch.abs(fvc_true - fvc_pred)

        # Handle error thresholding
        if not self.for_training:
            # For evaluation/validation, clip error at 1000ml as per metric def
            delta = torch.clamp(delta, max=self.max_error)

        # Calculate Loss (Negative Log Likelihood)
        # term1 = (sqrt(2) * delta) / sigma
        term1 = (self.sqrt_2 * delta) / sigma_clipped

        # term2 = ln(sqrt(2) * sigma)
        term2 = torch.log(self.sqrt_2 * sigma_clipped)

        loss = term1 + term2

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def compute_metric_score(fvc_pred, fvc_true, sigma_pred):
    """
    Calculates the exact competition metric score (negative value, higher is better).
    This function strictly follows the metric definition including all clipping rules.

    Returns:
        float: The mean metric score.
    """
    # Ensure inputs are tensors
    if not isinstance(fvc_pred, torch.Tensor):
        fvc_pred = torch.tensor(fvc_pred)
    if not isinstance(fvc_true, torch.Tensor):
        fvc_true = torch.tensor(fvc_true)
    if not isinstance(sigma_pred, torch.Tensor):
        sigma_pred = torch.tensor(sigma_pred)

    device = fvc_pred.device
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=device))

    fvc_pred = fvc_pred.view(-1)
    fvc_true = fvc_true.view(-1)
    sigma_pred = sigma_pred.view(-1)

    # Apply clipping
    sigma_clipped = torch.clamp(sigma_pred, min=Config.SIGMA_CLIP)
    delta = torch.abs(fvc_true - fvc_pred)
    delta = torch.clamp(delta, max=Config.MAX_ERROR)

    # Calculate metric
    metric = -((sqrt_2 * delta) / sigma_clipped + torch.log(sqrt_2 * sigma_clipped))

    return metric.mean().item()
