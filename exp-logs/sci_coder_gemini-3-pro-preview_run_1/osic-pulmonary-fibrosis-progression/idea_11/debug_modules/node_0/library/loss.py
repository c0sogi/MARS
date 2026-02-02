import torch
import torch.nn as nn
from library.config import Training


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the modified Laplace Log Likelihood loss function for the lung decline task.

    The competition metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    To serve as a Loss function for minimization, we negate the metric:
        Loss = (sqrt(2) * delta / sigma_clipped) + ln(sqrt(2) * sigma_clipped)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        # Load constants from configuration to ensure consistency
        self.min_sigma = Training.MIN_SIGMA_CLIP
        self.max_error = Training.MAX_ERROR_CLIP

        # Precompute sqrt(2) as a buffer to avoid recomputing every forward pass
        self.register_buffer("sqrt_2", torch.sqrt(torch.tensor(2.0)))

    def forward(self, pred_fvc, pred_sigma, true_fvc):
        """
        Calculates the loss.

        Args:
            pred_fvc (torch.Tensor): Predicted FVC values (Batch, 1).
            pred_sigma (torch.Tensor): Predicted Confidence/Sigma values (Batch, 1).
            true_fvc (torch.Tensor): Ground truth FVC values (Batch, 1).

        Returns:
            torch.Tensor: Scalar loss value (mean over batch).
        """
        # 1. Clip the confidence values (sigma)
        # We use clamp_min to enforce the lower bound of 70ml
        sigma_clipped = torch.clamp(pred_sigma, min=self.min_sigma)

        # 2. Calculate the absolute error
        abs_error = torch.abs(true_fvc - pred_fvc)

        # 3. Clip the error (delta)
        # We use clamp_max to enforce the upper bound of 1000ml
        # This prevents outliers from causing exploding gradients
        delta = torch.clamp(abs_error, max=self.max_error)

        # 4. Compute the loss components
        # Term 1: Scaled error
        term1 = (self.sqrt_2 * delta) / sigma_clipped

        # Term 2: Log uncertainty
        # ln(sqrt(2) * sigma) = ln(sqrt(2)) + ln(sigma)
        term2 = torch.log(self.sqrt_2 * sigma_clipped)

        # 5. Combine terms
        # Loss = Term1 + Term2
        loss = term1 + term2

        # Return mean loss over the batch
        return torch.mean(loss)
