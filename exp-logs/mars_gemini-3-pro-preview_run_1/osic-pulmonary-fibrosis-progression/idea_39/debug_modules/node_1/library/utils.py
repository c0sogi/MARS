import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Custom Loss function implementing the modified Laplace Log Likelihood.

    The competition metric is defined as:
        Metric = - (sqrt(2) * Delta) / Sigma_clipped - ln(sqrt(2) * Sigma_clipped)

    Where:
        Delta = min(|True - Pred|, 1000)
        Sigma_clipped = max(Sigma, 70)

    Since we want to maximize the Metric, we minimize the Loss:
        Loss = -Metric = (sqrt(2) * Delta) / Sigma_clipped + ln(sqrt(2) * Sigma_clipped)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.max_error = Config.MAX_ERROR
        self.min_confidence = Config.MIN_CONFIDENCE
        self.sqrt_2 = math.sqrt(2)

    def forward(self, pred_fvc, pred_sigma, target_fvc):
        """
        Computes the loss for a batch of predictions.

        Args:
            pred_fvc (torch.Tensor): Predicted FVC values.
            pred_sigma (torch.Tensor): Predicted Confidence (Sigma) values.
            target_fvc (torch.Tensor): Ground truth FVC values.

        Returns:
            torch.Tensor: The mean loss over the batch.
        """
        # Ensure correct shapes for broadcasting
        if pred_fvc.shape != target_fvc.shape:
            target_fvc = target_fvc.view_as(pred_fvc)

        # Calculate absolute error
        abs_error = torch.abs(target_fvc - pred_fvc)

        # Apply error thresholding (Delta)
        # "The error is thresholded at 1000 ml"
        delta = torch.clamp(abs_error, max=self.max_error)

        # Apply confidence clipping (Sigma_clipped)
        # "confidence values are clipped at 70 ml"
        sigma_clipped = torch.clamp(pred_sigma, min=self.min_confidence)

        # Calculate the two terms of the loss
        # Term 1: (sqrt(2) * Delta) / Sigma_clipped
        term1 = (self.sqrt_2 * delta) / sigma_clipped

        # Term 2: ln(sqrt(2) * Sigma_clipped)
        # Note: torch.log is natural logarithm (ln)
        term2 = torch.log(self.sqrt_2 * sigma_clipped)

        # Sum terms to get negative metric (Loss)
        loss = term1 + term2

        return torch.mean(loss)


def calculate_metric(y_true, y_pred, y_conf):
    """
    Calculates the official competition metric on numpy arrays.
    Used for validation and evaluation scoring.

    Args:
        y_true (np.array): Ground truth FVC values.
        y_pred (np.array): Predicted FVC values.
        y_conf (np.array): Predicted Confidence values.

    Returns:
        float: The mean metric score (higher is better).
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_conf = np.array(y_conf)

    # Calculate absolute error
    abs_error = np.abs(y_true - y_pred)

    # Apply error thresholding
    delta = np.minimum(abs_error, Config.MAX_ERROR)

    # Apply confidence clipping
    sigma_clipped = np.maximum(y_conf, Config.MIN_CONFIDENCE)

    # Calculate metric
    sqrt_2 = math.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric)
