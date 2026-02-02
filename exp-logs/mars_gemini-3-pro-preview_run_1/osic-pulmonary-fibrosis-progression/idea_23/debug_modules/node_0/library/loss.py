import torch
import torch.nn as nn
import math
from library.config import Config


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the modified Laplace Log Likelihood loss function.

    Metric Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Loss:
        Loss = -Metric (since we want to maximize the metric)
    """

    def __init__(self):
        super().__init__()
        self.sigma_clip = Config.SIGMA_CLIP
        self.error_clip = Config.ERROR_CLIP
        self.sqrt_2 = math.sqrt(2)

    def forward(self, pred_fvc, pred_sigma, true_fvc):
        """
        Args:
            pred_fvc (torch.Tensor): Predicted FVC values (Batch_Size,)
            pred_sigma (torch.Tensor): Predicted Confidence/Sigma values (Batch_Size,)
            true_fvc (torch.Tensor): Ground truth FVC values (Batch_Size,)

        Returns:
            loss (torch.Tensor): Scalar loss value to minimize.
        """
        # Ensure inputs are float for calculation
        pred_fvc = pred_fvc.float()
        pred_sigma = pred_sigma.float()
        true_fvc = true_fvc.float()

        # 1. Calculate Absolute Error (Delta)
        delta = torch.abs(true_fvc - pred_fvc)

        # 2. Clip Error (Delta) at 1000ml
        # This prevents large outliers from dominating the gradient
        delta_clipped = torch.clamp(delta, max=self.error_clip)

        # 3. Clip Sigma (Confidence) at 70ml
        # This reflects approximate measurement uncertainty
        sigma_clipped = torch.clamp(pred_sigma, min=self.sigma_clip)

        # 4. Calculate Metric Terms
        # Term 1: - (sqrt(2) * delta) / sigma
        term1 = (self.sqrt_2 * delta_clipped) / sigma_clipped

        # Term 2: - ln(sqrt(2) * sigma)
        term2 = torch.log(self.sqrt_2 * sigma_clipped)

        # Combine to get the metric (negative log likelihood)
        # The metric defined in the task is negative, higher is better.
        # metric = - term1 - term2
        metric = -term1 - term2

        # 5. Return Loss
        # We want to MAXIMIZE the metric, so we MINIMIZE the negative metric.
        # Loss = -1 * metric = term1 + term2
        loss = -torch.mean(metric)

        return loss
