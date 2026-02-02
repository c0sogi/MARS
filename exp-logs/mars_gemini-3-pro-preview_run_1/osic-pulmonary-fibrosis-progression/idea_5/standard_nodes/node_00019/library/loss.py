import torch
import torch.nn as nn
import numpy as np


class ModifiedLaplaceLoss(nn.Module):
    """
    Calculates the negative Modified Laplace Log Likelihood loss.

    This loss function is designed to optimize the competition metric directly.
    It handles the parametric predictions (slope and uncertainty components)
    and applies the specific clipping rules defined in the metric.

    Formula:
        Loss = - Metric
        Metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
        Loss = (sqrt(2) * delta) / sigma_clipped + ln(sqrt(2) * sigma_clipped)

    Where:
        delta = min(|True_FVC - Pred_FVC|, 1000)
        sigma_clipped = max(Pred_Sigma, 70)
    """

    def __init__(self):
        super(ModifiedLaplaceLoss, self).__init__()
        # Register sqrt(2) as a buffer or constant to avoid recomputing
        self.sqrt_2 = np.sqrt(2)

    def forward(self, alpha, sigma_base, sigma_growth, time, baseline_fvc, target_fvc):
        """
        Args:
            alpha (torch.Tensor): Predicted slope of decline. Shape (B,)
            sigma_base (torch.Tensor): Predicted base confidence. Shape (B,)
            sigma_growth (torch.Tensor): Predicted uncertainty growth. Shape (B,)
            time (torch.Tensor): Time delta (Week - Baseline_Week). Shape (B,)
            baseline_fvc (torch.Tensor): Patient's baseline FVC. Shape (B,)
            target_fvc (torch.Tensor): Ground truth FVC at the specific week. Shape (B,)

        Returns:
            torch.Tensor: Scalar loss value (mean over batch).
        """
        # 1. Reconstruct Predictions based on Linear Trajectory Model
        # FVC_pred = FVC_base + alpha * t
        pred_fvc = baseline_fvc + alpha * time

        # Sigma_pred = Sigma_base + Sigma_growth * |t|
        # Note: sigma_base and sigma_growth are already ensured positive by Softplus in the model
        pred_sigma = sigma_base + sigma_growth * torch.abs(time)

        # 2. Calculate Delta (Absolute Error) with Clipping
        # delta = min(|FVC_true - FVC_pred|, 1000)
        abs_error = torch.abs(target_fvc - pred_fvc)
        delta = torch.clamp(abs_error, max=1000.0)

        # 3. Calculate Sigma with Clipping
        # sigma_clipped = max(sigma, 70)
        sigma_clipped = torch.clamp(pred_sigma, min=70.0)

        # 4. Calculate Negative Log Likelihood components
        # Original Metric: - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
        # We want to maximize Metric, so we minimize Loss = -Metric

        # Term 1: (sqrt(2) * delta) / sigma_clipped
        term1 = (self.sqrt_2 * delta) / sigma_clipped

        # Term 2: ln(sqrt(2) * sigma_clipped)
        term2 = torch.log(self.sqrt_2 * sigma_clipped)

        # Sum terms to get Loss
        loss = term1 + term2

        # Return mean loss over the batch
        return torch.mean(loss)
