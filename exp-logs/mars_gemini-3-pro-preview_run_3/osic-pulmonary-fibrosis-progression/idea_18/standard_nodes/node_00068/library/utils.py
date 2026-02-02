import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the Metric-Aligned Laplace Log Likelihood Loss.
    Formula: L = (sqrt(2) * |y_true - y_pred|) / sigma + log(sqrt(2) * sigma)

    Expects predictions in standardized space. Does not enforce the 70ml clip
    during training to allow natural gradient flow for uncertainty estimation.
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        # Register sqrt(2) as a buffer to avoid recomputing it every forward pass
        self.register_buffer("sqrt_2", torch.sqrt(torch.tensor(2.0)))

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Model output of shape (Batch, 2).
                                  Column 0: Predicted Mean (mu).
                                  Column 1: Raw Confidence Logits (pre-softplus).
            targets (torch.Tensor): True targets of shape (Batch) or (Batch, 1).

        Returns:
            torch.Tensor: Scalar mean loss.
        """
        # Ensure targets are (Batch, 1)
        if targets.dim() == 1:
            targets = targets.view(-1, 1)

        mu = preds[:, 0:1]
        raw_sigma = preds[:, 1:2]

        # Enforce positivity for sigma using softplus + epsilon for numerical stability
        sigma = F.softplus(raw_sigma) + 1e-6

        # Calculate the loss components
        absolute_error = torch.abs(targets - mu)
        loss = (self.sqrt_2 * absolute_error) / sigma + torch.log(self.sqrt_2 * sigma)

        return torch.mean(loss)


def inverse_transform(mu_scaled, sigma_scaled):
    """
    Converts standardized model outputs (Z-scores) back to the original ml scale.

    Args:
        mu_scaled (np.ndarray or float): Predicted FVC in standardized space.
        sigma_scaled (np.ndarray or float): Predicted Confidence in standardized space.

    Returns:
        tuple: (mu_original, sigma_original) in ml units.
    """
    # Reverse Z-score standardization for the mean
    mu_original = mu_scaled * Config.TARGET_STD + Config.TARGET_MEAN

    # Reverse scaling for the standard deviation (scale only, no shift)
    sigma_original = sigma_scaled * Config.TARGET_STD

    return mu_original, sigma_original


def calculate_metric(y_true, y_pred, sigma):
    """
    Computes the competition metric: Modified Laplace Log Likelihood.

    Rules:
    1. sigma_clipped = max(sigma, 70)
    2. delta = min(|y_true - y_pred|, 1000)
    3. metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.ndarray): Ground truth FVC (ml).
        y_pred (np.ndarray): Predicted FVC (ml).
        sigma (np.ndarray): Predicted Confidence (ml).

    Returns:
        float: The mean metric score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sigma = np.array(sigma)

    # Apply competition-specific clipping
    sigma_clipped = np.maximum(sigma, Config.CONFIDENCE_CLIP)

    # Calculate absolute error and clip it at 1000 ml
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, Config.MAX_ERROR_CLIP)

    # Compute the metric
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta / sigma_clipped) - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric)
