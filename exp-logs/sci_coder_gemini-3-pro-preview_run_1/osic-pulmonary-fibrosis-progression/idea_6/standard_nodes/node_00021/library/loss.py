import torch
import torch.nn as nn
from library.config import Config


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the modified Laplace Log Likelihood loss function for the OSIC Pulmonary Fibrosis Progression task.

    The metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    We minimize the Negative Log Likelihood, which corresponds to maximizing the metric.
    Loss = -metric = (sqrt(2) * delta / sigma_clipped) + ln(sqrt(2) * sigma_clipped)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.min_sigma = Config.min_sigma
        self.max_delta = Config.max_delta
        # Register sqrt(2) as a buffer so it automatically moves to the correct device
        self.register_buffer("sqrt_2", torch.sqrt(torch.tensor(2.0)))

    def forward(self, preds, target_fvc, time_delta, baseline_fvc):
        """
        Compute the loss.

        Args:
            preds (torch.Tensor): Model predictions of shape (Batch, 3).
                                  Columns are [alpha, sigma_base, sigma_growth].
            target_fvc (torch.Tensor): True FVC values of shape (Batch, 1) or (Batch,).
            time_delta (torch.Tensor): Time elapsed since baseline (Weeks) of shape (Batch, 1) or (Batch,).
            baseline_fvc (torch.Tensor): Baseline FVC values of shape (Batch, 1) or (Batch,).

        Returns:
            torch.Tensor: Scalar loss value (mean over batch).
        """
        # Ensure inputs are column vectors for consistent broadcasting
        if target_fvc.ndim == 1:
            target_fvc = target_fvc.view(-1, 1)
        if time_delta.ndim == 1:
            time_delta = time_delta.view(-1, 1)
        if baseline_fvc.ndim == 1:
            baseline_fvc = baseline_fvc.view(-1, 1)

        # Unpack predictions
        # alpha: Slope of FVC decline/incline
        # sigma_base: Confidence at t=0
        # sigma_growth: Growth of uncertainty over time
        alpha = preds[:, 0:1]
        sigma_base = preds[:, 1:2]
        sigma_growth = preds[:, 2:3]

        # 1. Predict FVC: FVC_pred = Baseline + alpha * t
        fvc_pred = baseline_fvc + alpha * time_delta

        # 2. Predict Confidence (Sigma): Sigma = Sigma_base + Sigma_growth * |t|
        # Note: We calculate the raw sigma linear combination.
        # The clipping operation max(sigma, 70) ensures the metric's validity.
        sigma = sigma_base + sigma_growth * torch.abs(time_delta)

        # Apply clipping to Sigma (min 70 ml)
        sigma_clipped = torch.clamp(sigma, min=self.min_sigma)

        # 3. Calculate Error (Delta)
        # Delta = min(|True - Pred|, 1000)
        delta = torch.abs(target_fvc - fvc_pred)
        delta_clipped = torch.clamp(delta, max=self.max_delta)

        # 4. Compute Loss components
        # Loss = (sqrt(2) * delta_clipped) / sigma_clipped + ln(sqrt(2) * sigma_clipped)
        term1 = (self.sqrt_2 * delta_clipped) / sigma_clipped
        term2 = torch.log(self.sqrt_2 * sigma_clipped)

        loss = term1 + term2

        return loss.mean()

    def metric(self, preds, target_fvc, time_delta, baseline_fvc):
        """
        Calculates the competition metric (negative loss) for validation/logging.
        Values will be negative; higher (closer to 0) is better.
        """
        with torch.no_grad():
            loss = self.forward(preds, target_fvc, time_delta, baseline_fvc)
            return -loss
