import torch
import torch.nn as nn
from library.config import Config


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the modified Laplace Log Likelihood loss function.
    Optimizes the competition metric directly by minimizing the negative metric score.

    Formula:
        Metric = - (sqrt(2) * Delta_clipped) / Sigma_clipped - ln(sqrt(2) * Sigma_clipped)
        Loss   = - Metric
               = (sqrt(2) * Delta_clipped) / Sigma_clipped + ln(sqrt(2) * Sigma_clipped)

    Where:
        Delta_clipped = min(|True_FVC - Pred_FVC|, 1000)
        Sigma_clipped = max(Pred_Sigma, 70)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        # Register sqrt(2) as a buffer so it moves with the model device
        self.register_buffer("sqrt_2", torch.sqrt(torch.tensor(2.0)))

    def forward(self, outputs, targets, meta):
        """
        Args:
            outputs (torch.Tensor): Shape (B, 3). Contains [alpha, sigma_base, sigma_growth].
            targets (torch.Tensor): Shape (B,). True FVC values.
            meta (torch.Tensor): Shape (B, 2). Contains [Baseline_FVC, Delta_Week].

        Returns:
            torch.Tensor: Scalar loss value (mean over batch).
        """
        # 1. Unpack Model Outputs
        # alpha: Slope of decline/improvement
        # sigma_base: Uncertainty at week 0
        # sigma_growth: Uncertainty accumulation over time
        alpha = outputs[:, 0]
        sigma_base = outputs[:, 1]
        sigma_growth = outputs[:, 2]

        # 2. Unpack Metadata
        base_fvc = meta[:, 0]
        dt = meta[:, 1]

        # 3. Reconstruct Predictions based on Parametric Model
        # FVC = Base + Slope * Time
        pred_fvc = base_fvc + alpha * dt

        # Sigma = Base + Growth * |Time|
        pred_sigma = sigma_base + sigma_growth * torch.abs(dt)

        # 4. Calculate Error Components
        true_fvc = targets
        delta = torch.abs(true_fvc - pred_fvc)

        # 5. Apply Metric Clipping
        # "The error is thresholded at 1000 ml"
        delta_clipped = torch.clamp(delta, max=Config.MAX_ERROR)

        # "Confidence values are clipped at 70 ml"
        sigma_clipped = torch.clamp(pred_sigma, min=Config.MIN_CONFIDENCE)

        # 6. Calculate Loss
        # Loss = (sqrt(2) * delta) / sigma + ln(sqrt(2) * sigma)
        term1 = (self.sqrt_2 * delta_clipped) / sigma_clipped
        term2 = torch.log(self.sqrt_2 * sigma_clipped)

        loss = term1 + term2

        return torch.mean(loss)
