import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LaplaceNLLLoss(nn.Module):
    """
    Implements the modified Laplace Log Likelihood loss function for training.

    This loss function corresponds to the Negative Log Likelihood (NLL) of a Laplace
    distribution. It is designed to minimize the error in the standardized space
    without the hard clipping thresholds used in the competition evaluation metric,
    thereby ensuring non-zero gradients for optimization.

    Formula:
        L = (sqrt(2) * |y_true - mu|) / sigma + ln(sqrt(2) * sigma)

    Where:
        sigma = softplus(raw_sigma) + epsilon
    """

    def __init__(self, epsilon=1e-6):
        """
        Args:
            epsilon (float): Small constant to prevent division by zero or log(0).
        """
        super(LaplaceNLLLoss, self).__init__()
        self.epsilon = epsilon
        # Precompute sqrt(2) as a constant
        self.sqrt_2 = np.sqrt(2)

    def forward(self, pred, target):
        """
        Calculates the Laplace NLL loss.

        Args:
            pred (torch.Tensor): Predicted output of shape (Batch, 2).
                                 - pred[:, 0]: Predicted Mean (mu), standardized.
                                 - pred[:, 1]: Predicted Raw Sigma (log-scale-ish), standardized.
            target (torch.Tensor): Ground truth target of shape (Batch, 1) or (Batch,).
                                   Values should be standardized (Z-scored).

        Returns:
            torch.Tensor: Scalar loss value (averaged over the batch).
        """
        # Separate the mean and the raw sigma output
        pred_mean = pred[:, 0]
        pred_raw_sigma = pred[:, 1]

        # Ensure target shape matches pred_mean for broadcasting
        if target.dim() > 1:
            target = target.view(-1)

        if pred_mean.dim() > 1:
            pred_mean = pred_mean.view(-1)

        if pred_raw_sigma.dim() > 1:
            pred_raw_sigma = pred_raw_sigma.view(-1)

        # Apply softplus to enforce positive standard deviation
        # Adding epsilon prevents sigma from being exactly 0
        sigma = F.softplus(pred_raw_sigma) + self.epsilon

        # Calculate the absolute error |y - mu|
        abs_diff = torch.abs(target - pred_mean)

        # Calculate the first term: (sqrt(2) * |y - mu|) / sigma
        term1 = (self.sqrt_2 * abs_diff) / sigma

        # Calculate the second term: ln(sqrt(2) * sigma)
        # We compute this as log(sqrt(2) * sigma)
        term2 = torch.log(self.sqrt_2 * sigma)

        # Sum the terms to get the NLL
        loss = term1 + term2

        # Return the mean loss over the batch
        return torch.mean(loss)
