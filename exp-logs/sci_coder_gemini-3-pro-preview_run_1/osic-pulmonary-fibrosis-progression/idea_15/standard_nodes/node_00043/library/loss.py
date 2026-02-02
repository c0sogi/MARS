import torch
import torch.nn as nn
import numpy as np
from library.config import Config


class RobustLaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the Robust Laplace Log Likelihood Loss.

    This loss function is the negative of the competition metric. It incorporates
    specific clipping mechanisms for both the prediction error and the confidence
    estimates to handle outliers and enforce uncertainty constraints directly
    within the optimization process.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        loss = (sqrt(2) * delta) / sigma_clipped + ln(sqrt(2) * sigma_clipped)
    """

    def __init__(self):
        super().__init__()
        self.q_clip = Config.Q_CLIP
        self.max_err = Config.MAX_ERR
        # Precompute sqrt(2)
        self.sqrt_2 = np.sqrt(2)

    def forward(self, fvc_pred, confidence_pred, target):
        """
        Calculates the mean loss for the batch.

        Args:
            fvc_pred (torch.Tensor): Predicted FVC values of shape (B,).
            confidence_pred (torch.Tensor): Predicted Confidence (Sigma) values of shape (B,).
            target (torch.Tensor): Ground truth FVC values of shape (B,).

        Returns:
            torch.Tensor: The scalar loss value.
        """
        # Ensure constant is on the correct device
        sqrt_2 = torch.tensor(self.sqrt_2, device=fvc_pred.device, dtype=fvc_pred.dtype)

        # 1. Clip the confidence (sigma)
        # The metric requires sigma to be at least 70 ml.
        # confidence_pred is guaranteed positive by Softplus in the model,
        # but we must enforce the lower bound of 70.
        sigma_clipped = torch.clamp(confidence_pred, min=self.q_clip)

        # 2. Calculate the robust error (Delta)
        # Calculate absolute error
        abs_error = torch.abs(target - fvc_pred)
        # Threshold the error at 1000 ml.
        # This prevents large outliers from dominating the gradient.
        delta = torch.clamp(abs_error, max=self.max_err)

        # 3. Compute Loss Components
        # Term 1: Scaled Error
        term1 = (sqrt_2 * delta) / sigma_clipped

        # Term 2: Log Sigma Penalty
        term2 = torch.log(sqrt_2 * sigma_clipped)

        # Total Loss
        loss = term1 + term2

        # Return mean over the batch
        return torch.mean(loss)
