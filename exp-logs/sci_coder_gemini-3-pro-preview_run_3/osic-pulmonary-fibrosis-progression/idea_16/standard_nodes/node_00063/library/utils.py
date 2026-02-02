import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for various libraries to ensure reproducibility.
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


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Custom Loss function based on the modified Laplace Log Likelihood.
    Optimizes the network to maximize the competition metric.

    Loss = (sqrt(2) * |True - Pred|) / Sigma + ln(sqrt(2) * Sigma)

    Note: The metric clips Sigma at 70 and Delta at 1000.
    This loss function does NOT clip Delta to allow gradients for large errors,
    and uses softplus+epsilon for Sigma to ensure stability and differentiability.
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.epsilon = Config.EPSILON

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Shape (Batch, 2).
                                  Column 0 is FVC prediction (mu).
                                  Column 1 is Raw Confidence output (before activation).
            targets (torch.Tensor): Shape (Batch) or (Batch, 1). True FVC values.
        """
        # Separate FVC prediction and Raw Sigma
        fvc_pred = preds[:, 0]
        raw_sigma = preds[:, 1]

        # Ensure targets are the correct shape
        if targets.ndim > 1:
            targets = targets.squeeze()

        # Apply Softplus to ensure Sigma is positive
        # We do not clip at 70 here to allow the model to learn natural uncertainty
        sigma = F.softplus(raw_sigma) + self.epsilon

        # Calculate absolute error (Delta)
        delta = torch.abs(targets - fvc_pred)

        # Calculate Loss
        # We use the negative of the metric formula to create a minimization objective
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=preds.device))

        term1 = (sqrt_2 * delta) / sigma
        term2 = torch.log(sqrt_2 * sigma)

        loss = term1 + term2

        return torch.mean(loss)


def metric_score(y_true, y_pred, sigma_pred):
    """
    Calculates the official competition metric.

    Metric = - (sqrt(2) * Delta) / Sigma_clipped - ln(sqrt(2) * Sigma_clipped)

    Where:
        Sigma_clipped = max(Sigma, 70)
        Delta = min(|True - Pred|, 1000)

    Args:
        y_true (np.ndarray): True FVC values.
        y_pred (np.ndarray): Predicted FVC values.
        sigma_pred (np.ndarray): Predicted Confidence (Sigma) values.

    Returns:
        float: The mean metric score over the input arrays.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sigma_pred = np.array(sigma_pred)

    # Clip Sigma at 70ml
    sigma_clipped = np.maximum(sigma_pred, Config.METRIC_CLIP_SIGMA)

    # Calculate Delta and clip at 1000ml
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, Config.METRIC_MAX_ERROR)

    # Calculate Metric
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric)
