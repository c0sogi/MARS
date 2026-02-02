import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the negative log likelihood loss for a Laplace distribution.
    Formula: L = |y - mu|/sigma + ln(sigma)

    Expects the model to output [mu, raw_sigma].
    Applies softplus to raw_sigma to ensure positivity.
    """

    def __init__(self, epsilon=1e-6):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, preds, targets):
        """
        Computes the loss.

        Args:
            preds: Tensor of shape (Batch, 2) containing [mu, raw_sigma]
            targets: Tensor of shape (Batch,) or (Batch, 1) containing true FVC

        Returns:
            torch.Tensor: Scalar loss value (mean over batch).
        """
        # Extract mu and raw_sigma
        mu = preds[:, 0]
        raw_sigma = preds[:, 1]

        # Enforce positivity for sigma using softplus as per Idea/Training requirements
        sigma = F.softplus(raw_sigma) + self.epsilon

        # Flatten targets if necessary to match mu shape
        if targets.ndim > 1:
            targets = targets.view(-1)

        # Compute element-wise loss
        # Formula requested: |y - mu|/sigma + ln(sigma)
        loss = torch.abs(targets - mu) / sigma + torch.log(sigma)

        # Return mean loss
        return torch.mean(loss)


def calculate_metric(y_true, y_pred, sigma_pred):
    """
    Computes the competition-specific metric (Modified Laplace Log Likelihood).

    Metric = - (sqrt(2) * Delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)
    where:
        sigma_clipped = max(sigma, 70)
        Delta = min(|y_true - y_pred|, 1000)

    Args:
        y_true: Ground truth FVC (numpy array or tensor)
        y_pred: Predicted FVC (numpy array or tensor)
        sigma_pred: Predicted Confidence (numpy array or tensor)

    Returns:
        float: The mean metric score.
    """
    # Convert tensors to numpy if passed as tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma_pred, torch.Tensor):
        sigma_pred = sigma_pred.detach().cpu().numpy()

    # Flatten arrays to ensure 1D alignment
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    sigma_pred = sigma_pred.flatten()

    # Apply metric constraints defined in Config/Task
    sigma_clipped = np.maximum(sigma_pred, Config.MIN_CONFIDENCE)

    abs_error = np.abs(y_true - y_pred)
    delta = np.minimum(abs_error, Config.MAX_ERROR_METRIC)

    # Compute metric
    sqrt_2 = np.sqrt(2)
    metric_values = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric_values)
