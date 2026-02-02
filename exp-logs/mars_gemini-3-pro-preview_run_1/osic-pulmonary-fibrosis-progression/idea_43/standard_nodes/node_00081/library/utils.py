import torch
import torch.nn as nn
import numpy as np
import os
import random
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility using the Config's method.
    """
    Config.set_seed(seed)


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the modified Laplace Log Likelihood Loss.

    The competition metric is defined as:
    Metric = - (sqrt(2) * Delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Where:
    sigma_clipped = max(sigma, 70)
    Delta = min(|True - Pred|, 1000)

    Since we want to maximize the Metric, we minimize the Loss:
    Loss = -Metric
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.max_error = float(Config.MAX_ERROR)
        self.min_sigma = float(Config.MIN_SIGMA)
        # Register sqrt(2) as a buffer so it moves with the model to GPU
        self.register_buffer("sqrt_2", torch.sqrt(torch.tensor(2.0)))

    def forward(self, fvc_pred, sigma_pred, target):
        """
        Args:
            fvc_pred (torch.Tensor): Predicted FVC values.
            sigma_pred (torch.Tensor): Predicted Confidence (Sigma) values.
            target (torch.Tensor): Ground truth FVC values.

        Returns:
            torch.Tensor: Scalar loss value (mean over batch).
        """
        # Clip sigma to satisfy the metric requirement (approximate measurement uncertainty)
        sigma_clipped = torch.clamp(sigma_pred, min=self.min_sigma)

        # Calculate absolute error
        abs_error = torch.abs(target - fvc_pred)

        # Clip error at 1000 ml to avoid large errors adversely penalizing results
        delta = torch.clamp(abs_error, max=self.max_error)

        # Calculate the negative log likelihood terms
        # Loss = (sqrt(2) * Delta) / sigma + ln(sqrt(2) * sigma)
        term1 = (self.sqrt_2 * delta) / sigma_clipped
        term2 = torch.log(self.sqrt_2 * sigma_clipped)

        loss = term1 + term2

        return torch.mean(loss)


def calculate_metric(fvc_true, fvc_pred, sigma_pred):
    """
    Calculates the exact competition metric for validation/evaluation.
    Handles both PyTorch tensors and NumPy arrays.

    Args:
        fvc_true: Ground truth FVC.
        fvc_pred: Predicted FVC.
        sigma_pred: Predicted Confidence.

    Returns:
        float: The average modified Laplace Log Likelihood score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(fvc_true, torch.Tensor):
        fvc_true = fvc_true.detach().cpu().numpy()
    if isinstance(fvc_pred, torch.Tensor):
        fvc_pred = fvc_pred.detach().cpu().numpy()
    if isinstance(sigma_pred, torch.Tensor):
        sigma_pred = sigma_pred.detach().cpu().numpy()

    # Metric logic
    sigma_clipped = np.maximum(sigma_pred, Config.MIN_SIGMA)
    abs_error = np.abs(fvc_true - fvc_pred)
    delta = np.minimum(abs_error, Config.MAX_ERROR)

    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric)
