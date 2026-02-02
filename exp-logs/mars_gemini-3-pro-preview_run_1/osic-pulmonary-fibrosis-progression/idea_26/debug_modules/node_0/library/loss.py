import torch
import torch.nn as nn
import numpy as np
from library.config import Config


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the modified Laplace Log Likelihood loss function for the DP-SDAN model.
    The objective is to minimize the negative of the competition metric.

    Competition Metric:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Loss Function (Minimization Objective):
        Loss = -metric = (sqrt(2) * delta / sigma_clipped) + ln(sqrt(2) * sigma_clipped)
    """

    def __init__(self):
        super().__init__()
        self.max_error = Config.MAX_ERROR
        self.min_sigma = Config.MIN_SIGMA

    def forward(
        self,
        alpha,
        sigma_base,
        sigma_growth,
        baseline_fvc,
        delta_week,
        target_fvc,
    ):
        """
        Computes the loss based on the model's trajectory parameters and ground truth.

        Args:
            alpha (torch.Tensor): Predicted slope of decline (Batch,).
            sigma_base (torch.Tensor): Predicted baseline uncertainty (Batch,).
            sigma_growth (torch.Tensor): Predicted uncertainty growth rate (Batch,).
            baseline_fvc (torch.Tensor): Patient's baseline FVC (Batch,).
            delta_week (torch.Tensor): Relative week number (Current - Baseline) (Batch,).
            target_fvc (torch.Tensor): Ground truth FVC (Batch,).

        Returns:
            torch.Tensor: Scalar mean loss value.
        """
        # Create constant on the correct device
        device = alpha.device
        sqrt_2 = torch.tensor(np.sqrt(2), device=device, dtype=alpha.dtype)

        # 1. Reconstruct Prediction Trajectories
        # FVC_pred = Baseline_FVC + alpha * (Week - Baseline_Week)
        pred_fvc = baseline_fvc + alpha * delta_week

        # Sigma_pred = Sigma_base + Sigma_growth * |Week - Baseline_Week|
        # Note: sigma_base and sigma_growth are enforced positive by Softplus in the model
        pred_sigma = sigma_base + sigma_growth * torch.abs(delta_week)

        # 2. Compute Metric Components

        # Absolute Error
        abs_error = torch.abs(target_fvc - pred_fvc)

        # Apply Error Clipping (Robustness against outliers)
        # "The error is thresholded at 1000 ml"
        delta = torch.clamp(abs_error, max=self.max_error)

        # Apply Confidence Clipping
        # "confidence values are clipped at 70 ml"
        sigma_clipped = torch.clamp(pred_sigma, min=self.min_sigma)

        # 3. Calculate Loss
        # Loss = (sqrt(2) * delta / sigma_clipped) + ln(sqrt(2) * sigma_clipped)
        term1 = (sqrt_2 * delta) / sigma_clipped
        term2 = torch.log(sqrt_2 * sigma_clipped)

        loss = term1 + term2

        return loss.mean()
