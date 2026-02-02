import torch
import torch.nn as nn
import numpy as np


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the modified Laplace Log Likelihood loss function as defined in the task.

    The metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    The loss to be minimized is the negative of this metric:
        loss = (sqrt(2) * delta / sigma_clipped) + ln(sqrt(2) * sigma_clipped)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()

    def forward(self, pred_fvc, pred_sigma, true_fvc):
        """
        Calculates the loss.

        Args:
            pred_fvc (torch.Tensor): Predicted FVC values. Shape (Batch_Size, 1) or (Batch_Size,).
            pred_sigma (torch.Tensor): Predicted Confidence (sigma) values. Shape (Batch_Size, 1) or (Batch_Size,).
            true_fvc (torch.Tensor): Ground truth FVC values. Shape (Batch_Size, 1) or (Batch_Size,).

        Returns:
            torch.Tensor: Scalar loss value (mean over the batch).
        """
        # Ensure inputs are flattened to matching shapes
        pred_fvc = pred_fvc.view(-1)
        pred_sigma = pred_sigma.view(-1)
        true_fvc = true_fvc.view(-1)

        # Calculate absolute error
        delta = torch.abs(pred_fvc - true_fvc)

        # Apply robustness clipping to the error (max 1000 ml)
        # This prevents outliers from dominating the gradient
        delta_clipped = torch.clamp(delta, max=1000.0)

        # Apply robustness clipping to the confidence (min 70 ml)
        # This reflects approximate measurement uncertainty
        sigma_clipped = torch.clamp(pred_sigma, min=70.0)

        # Calculate the metric terms
        # Term 1: Scaled absolute error
        term1 = (
            torch.sqrt(torch.tensor(2.0, device=pred_fvc.device)) * delta_clipped
        ) / sigma_clipped

        # Term 2: Log of scaled confidence
        term2 = torch.log(
            torch.sqrt(torch.tensor(2.0, device=pred_fvc.device)) * sigma_clipped
        )

        # Loss is the sum of terms (negative of the metric)
        loss = term1 + term2

        # Return the mean loss over the batch
        return torch.mean(loss)
