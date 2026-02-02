import torch
import torch.nn as nn
import numpy as np
from library.config import Config


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the custom loss function based on the Modified Laplace Log Likelihood metric.

    Metric Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true_fvc - pred_fvc|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Training Objective:
        Minimize Loss = -Metric
        Loss = (sqrt(2) * delta) / sigma_clipped + ln(sqrt(2) * sigma_clipped)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.sigma_clip = Config.SIGMA_CLIP
        self.max_error = Config.MAX_ERROR
        # Pre-compute sqrt(2) as a float
        self.sqrt_2 = np.sqrt(2)

    def forward(self, pred_fvc, pred_sigma, true_fvc):
        """
        Calculates the loss for a batch of predictions.

        Args:
            pred_fvc (torch.Tensor): Predicted FVC values (Batch,).
            pred_sigma (torch.Tensor): Predicted Confidence/Sigma values (Batch,).
            true_fvc (torch.Tensor): Ground truth FVC values (Batch,).

        Returns:
            torch.Tensor: Scalar loss value (mean over batch).
        """
        # Ensure inputs are on the same device and type
        # pred_fvc and pred_sigma are likely already on device with gradients
        # true_fvc might need to be ensured float32

        # 1. Clip Sigma (Confidence)
        # The metric requires sigma to be at least 70 ml.
        # We use clamp to enforce this. Gradients will be 0 for values < 70,
        # effectively ignoring updates for sigma if it tries to go below the threshold.
        sigma_clipped = torch.clamp(pred_sigma, min=self.sigma_clip)

        # 2. Calculate Delta (Absolute Error)
        abs_error = torch.abs(true_fvc - pred_fvc)

        # 3. Clip Delta
        # The metric thresholds error at 1000 ml to avoid penalizing outliers too heavily.
        delta = torch.clamp(abs_error, max=self.max_error)

        # 4. Compute Loss components
        # We use a tensor for sqrt_2 to ensure device compatibility
        sqrt_2_t = torch.tensor(
            self.sqrt_2, device=pred_fvc.device, dtype=pred_fvc.dtype
        )

        # Term 1: (sqrt(2) * delta) / sigma_clipped
        term1 = (sqrt_2_t * delta) / sigma_clipped

        # Term 2: ln(sqrt(2) * sigma_clipped)
        term2 = torch.log(sqrt_2_t * sigma_clipped)

        # Total Loss = Term1 + Term2
        loss = term1 + term2

        # Return mean loss over the batch
        return torch.mean(loss)
