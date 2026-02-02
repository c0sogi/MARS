import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import inverse_transform, calculate_metric


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Metric-Aligned Laplace Log Likelihood Loss (Standardized).

    Optimizes the negative log likelihood of a Laplace distribution in the standardized space.
    This loss function is designed to align with the competition metric while operating
    on standardized data to ensure training stability.

    Formula: L = (sqrt(2) * |y_true - y_pred|) / sigma + ln(sqrt(2) * sigma)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        # Precompute constants
        self.sqrt_2 = np.sqrt(2.0)
        self.log_sqrt_2 = np.log(self.sqrt_2)

    def forward(self, mu_pred, sigma_pred, target):
        """
        Calculates the loss.

        Args:
            mu_pred (torch.Tensor): Predicted FVC mean (standardized).
            sigma_pred (torch.Tensor): Predicted FVC confidence/std (standardized).
            target (torch.Tensor): True FVC (standardized).

        Returns:
            torch.Tensor: The mean loss over the batch.
        """
        # Calculate absolute error (delta)
        delta = torch.abs(target - mu_pred)

        # Calculate Negative Log Likelihood
        # Note: The competition metric is negative (higher is better).
        # We minimize the positive NLL (lower is better).
        # Loss = (sqrt(2) * delta) / sigma + ln(sqrt(2)) + ln(sigma)
        loss = (
            (self.sqrt_2 * delta) / sigma_pred + torch.log(sigma_pred) + self.log_sqrt_2
        )

        return torch.mean(loss)


def competition_metric(mu_pred, sigma_pred, target):
    """
    Computes the official competition metric using standardized inputs.

    This function acts as a bridge between the model's standardized outputs and
    the official evaluation logic. It handles inverse scaling and delegates
    the final score calculation to the library utility.

    Args:
        mu_pred (torch.Tensor): Predicted FVC mean (standardized).
        sigma_pred (torch.Tensor): Predicted FVC confidence/std (standardized).
        target (torch.Tensor): True FVC (standardized).

    Returns:
        float: The average metric score (negative value, higher is better).
    """
    # 1. Inverse transform predictions to original scale (ml)
    # inverse_transform applies the competition requirement: max(sigma, 70)
    # It returns tensors or numpy arrays based on input; here inputs are likely tensors.
    fvc_pred_ml, sigma_pred_ml = inverse_transform(mu_pred, sigma_pred)

    # 2. Inverse transform target to original scale (ml)
    # The target was standardized using (FVC - Mean) / Std
    # So, FVC = target * Std + Mean
    if torch.is_tensor(target):
        fvc_true_ml = target * Config.TARGET_STD + Config.TARGET_MEAN
    else:
        fvc_true_ml = target * Config.TARGET_STD + Config.TARGET_MEAN

    # 3. Calculate the official metric
    # calculate_metric handles the error clipping: min(|true - pred|, 1000)
    # and computes the final Laplace Log Likelihood score.
    score = calculate_metric(fvc_true_ml, fvc_pred_ml, sigma_pred_ml)

    return score
