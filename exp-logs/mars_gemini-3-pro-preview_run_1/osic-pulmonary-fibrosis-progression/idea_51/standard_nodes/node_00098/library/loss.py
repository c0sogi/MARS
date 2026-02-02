import torch
import torch.nn as nn
from library.config import Config


class LaplaceLikelihoodLoss(nn.Module):
    """
    Custom Loss function that directly optimizes the competition metric:
    Modified Laplace Log Likelihood.

    The model outputs static parameters (alpha, sigma_base, sigma_growth),
    which are used to reconstruct the FVC and Confidence trajectories
    before calculating the loss.
    """

    def __init__(self):
        super().__init__()
        self.max_error = Config.METRIC_MAX_ERROR
        self.min_confidence = Config.METRIC_MIN_CONFIDENCE
        # Register sqrt(2) as a buffer to avoid recomputing it every forward pass
        self.register_buffer("sqrt_2", torch.sqrt(torch.tensor(2.0)))

    def forward(self, outputs, target, baseline_fvc, time_delta):
        """
        Args:
            outputs (torch.Tensor): Shape (B, 3). Contains [alpha, sigma_base, sigma_growth].
            target (torch.Tensor): Shape (B, 1). True FVC values at the specific week.
            baseline_fvc (torch.Tensor): Shape (B, 1). Baseline FVC values.
            time_delta (torch.Tensor): Shape (B, 1). (Week - Baseline_Week).

        Returns:
            torch.Tensor: Scalar loss value (mean over batch).
        """
        # 1. Unpack Model Outputs
        # alpha can be negative (decline) or positive (improvement)
        alpha = outputs[:, 0:1]
        # sigma_base and sigma_growth are enforced positive by Softplus in the model
        sigma_base = outputs[:, 1:2]
        sigma_growth = outputs[:, 2:3]

        # 2. Reconstruct Trajectories
        # FVC_pred = Baseline + alpha * dt
        fvc_pred = baseline_fvc + alpha * time_delta

        # Sigma_pred = Base + Growth * |dt|
        # Confidence grows with time distance from baseline
        sigma_pred = sigma_base + sigma_growth * torch.abs(time_delta)

        # 3. Apply Metric Constraints (Clipping)
        # Clip confidence at 70ml
        sigma_clipped = torch.clamp(sigma_pred, min=self.min_confidence)

        # Calculate absolute error
        abs_error = torch.abs(target - fvc_pred)

        # Clip error at 1000ml (Filter outliers)
        delta = torch.clamp(abs_error, max=self.max_error)

        # 4. Compute Negative Metric (Loss)
        # Metric = - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
        # Loss = -Metric = (sqrt(2) * delta) / sigma + ln(sqrt(2) * sigma)

        term1 = (self.sqrt_2 * delta) / sigma_clipped
        term2 = torch.log(self.sqrt_2 * sigma_clipped)

        loss = term1 + term2

        return torch.mean(loss)
