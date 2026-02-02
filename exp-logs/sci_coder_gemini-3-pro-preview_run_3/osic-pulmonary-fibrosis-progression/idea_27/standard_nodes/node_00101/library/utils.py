import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
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


class LaplaceLogLikelihood(nn.Module):
    """
    Metric-Aligned Laplace Log Likelihood Loss.
    Formula: L = (sqrt(2) * |y_true - y_pred|) / sigma + ln(sqrt(2) * sigma)

    This loss assumes that the input `preds` contains [FVC_pred, Sigma_pred]
    where Sigma_pred is already positive (e.g. after softplus).
    """

    def __init__(self):
        super(LaplaceLogLikelihood, self).__init__()
        # Register constants as buffers to ensure they are on the correct device
        self.register_buffer("sqrt_2", torch.sqrt(torch.tensor(2.0)))
        self.register_buffer("log_sqrt_2", torch.log(torch.sqrt(torch.tensor(2.0))))

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Shape (Batch, 2).
                                  preds[:, 0] -> Predicted FVC (mu)
                                  preds[:, 1] -> Predicted Confidence (sigma)
            targets (torch.Tensor): Shape (Batch,) or (Batch, 1). True FVC.

        Returns:
            torch.Tensor: Scalar loss value (mean over batch).
        """
        # Ensure targets are (Batch, 1)
        if targets.dim() == 1:
            targets = targets.view(-1, 1)

        fvc_pred = preds[:, 0].view(-1, 1)
        sigma_pred = preds[:, 1].view(-1, 1)

        # Calculate Delta: |y_true - y_pred|
        delta = torch.abs(targets - fvc_pred)

        # Term 1: (sqrt(2) * delta) / sigma
        # Adding a small epsilon to sigma for numerical stability,
        # though the model should enforce positivity via softplus + epsilon.
        term1 = (self.sqrt_2 * delta) / (sigma_pred + 1e-8)

        # Term 2: ln(sqrt(2) * sigma)
        term2 = self.log_sqrt_2 + torch.log(sigma_pred + 1e-8)

        loss = term1 + term2

        return torch.mean(loss)


def calculate_competition_metric(y_true, y_pred, sigma_pred):
    """
    Calculates the competition metric for evaluation.

    Metric logic:
    sigma_clipped = max(sigma, 70)
    delta = min(|FVC_true - FVC_pred|, 1000)
    metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.ndarray): True FVC values (ml).
        y_pred (np.ndarray): Predicted FVC values (ml).
        sigma_pred (np.ndarray): Predicted Confidence values (ml).

    Returns:
        float: The mean metric score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sigma_pred = np.array(sigma_pred)

    # Apply metric constraints
    sigma_clipped = np.maximum(sigma_pred, 70)
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, 1000)

    sqrt_2 = np.sqrt(2)

    # Compute metric
    metric = -(sqrt_2 * delta / sigma_clipped) - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric)


class TargetScaler:
    """
    Helper class for Z-score scaling and inverse scaling of the target variable (FVC).
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, values):
        """
        Compute mean and std from the training values.
        """
        self.mean = np.mean(values)
        self.std = np.std(values)

    def transform(self, values):
        """
        Apply Z-score scaling: (x - mean) / std
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted.")
        return (values - self.mean) / self.std

    def inverse_transform(self, values):
        """
        Inverse Z-score scaling for FVC: x * std + mean
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted.")
        return values * self.std + self.mean

    def inverse_transform_sigma(self, sigma_values):
        """
        Inverse scaling for Sigma (standard deviation).
        Since sigma represents a spread, it is scaled only by the std factor, not shifted by mean.
        sigma_original = sigma_scaled * std
        """
        if self.std is None:
            raise ValueError("Scaler has not been fitted.")
        return sigma_values * self.std
