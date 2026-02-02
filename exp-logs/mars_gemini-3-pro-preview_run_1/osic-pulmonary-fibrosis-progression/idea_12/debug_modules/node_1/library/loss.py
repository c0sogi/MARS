import torch
import torch.nn as nn
from library.config import Config


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the Modified Laplace Log Likelihood Loss for Pulmonary Fibrosis prediction.

    Optimizes the metric:
        Metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Where:
        delta = min(|True - Pred|, 1000)
        sigma_clipped = max(sigma, 70)

    The Loss is the negative of the Metric (to be minimized).
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.device = Config.DEVICE
        self.max_error = Config.MAX_ERROR_THRESHOLD
        self.min_confidence = Config.MIN_CONFIDENCE

        # Pre-calculate sqrt(2) on the correct device
        self.sqrt_2 = torch.sqrt(torch.tensor(2.0, device=self.device))

    def forward(self, preds, week_delta, baseline_fvc, target_fvc):
        """
        Calculates the loss for a batch.

        Args:
            preds (torch.Tensor): Output from the network of shape (B, 3).
                                  Columns: [Alpha, Sigma_Base, Sigma_Growth]
            week_delta (torch.Tensor): Time difference (Week - Baseline_Week) of shape (B,).
            baseline_fvc (torch.Tensor): FVC at week 0 of shape (B,).
            target_fvc (torch.Tensor): Ground truth FVC at current week of shape (B,).

        Returns:
            torch.Tensor: Scalar mean loss.
        """
        # 1. Unpack Predictions
        # alpha: Slope of decline (ml/week)
        # sigma_base: Uncertainty at baseline (ml)
        # sigma_growth: Uncertainty growth rate (ml/week)
        alpha = preds[:, 0]
        sigma_base = preds[:, 1]
        sigma_growth = preds[:, 2]

        # 2. Compute Predicted FVC
        # Linear trajectory: FVC = Baseline + Alpha * Delta_Time
        fvc_pred = baseline_fvc + alpha * week_delta

        # 3. Compute Predicted Confidence (Sigma)
        # Linear uncertainty: Sigma = Base + Growth * |Delta_Time|
        # Note: sigma_base and sigma_growth are already positive (Softplus in network)
        sigma_pred = sigma_base + sigma_growth * torch.abs(week_delta)

        # 4. Calculate Metric Components

        # Absolute Error
        abs_error = torch.abs(target_fvc - fvc_pred)

        # Error Clipping (Threshold at 1000ml)
        # Acts as a robust regression filter for large outliers
        delta = torch.clamp(abs_error, max=self.max_error)

        # Confidence Clipping (Threshold at 70ml)
        # Ensures metric stability and matches evaluation criteria
        sigma_clipped = torch.clamp(sigma_pred, min=self.min_confidence)

        # 5. Calculate Loss
        # Loss = (sqrt(2) * delta) / sigma + ln(sqrt(2) * sigma)
        term1 = (self.sqrt_2 * delta) / sigma_clipped
        term2 = torch.log(self.sqrt_2 * sigma_clipped)

        loss = term1 + term2

        return torch.mean(loss)
