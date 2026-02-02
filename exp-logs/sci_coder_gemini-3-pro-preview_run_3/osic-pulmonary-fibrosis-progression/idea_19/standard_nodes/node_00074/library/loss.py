import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class MetricAlignedLLLoss(nn.Module):
    """
    Implements the custom loss function required to address the 'Loss Mismatch'.
    Calculates the negative log likelihood assuming a Laplace distribution with
    the specific constants from the metric.

    Formula:
        loss = ln(sqrt(2) * sigma) + (sqrt(2) * abs(y_true - y_pred)) / sigma

    This loss allows gradients to flow through sigma without the hard clipping
    (70ml) used in the final evaluation metric.
    """

    def __init__(self, epsilon: float = 1e-6):
        super(MetricAlignedLLLoss, self).__init__()
        self.epsilon = epsilon
        self.metric_constant = Config.METRIC_CONSTANT  # sqrt(2)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the Metric-Aligned Laplace Log Likelihood Loss.

        Args:
            pred (torch.Tensor): Predictions of shape (batch_size, 2).
                - Column 0: Predicted FVC (mu)
                - Column 1: Raw confidence output (before softplus)
            target (torch.Tensor): True FVC values of shape (batch_size,) or (batch_size, 1).

        Returns:
            torch.Tensor: The mean loss over the batch.
        """
        # Ensure target shape is (batch_size, 1)
        if target.dim() == 1:
            target = target.view(-1, 1)

        # Extract predictions
        fvc_pred = pred[:, 0].view(-1, 1)
        raw_sigma = pred[:, 1].view(-1, 1)

        # Apply softplus to ensure positivity for sigma, add epsilon for stability
        # We do NOT clip to 70 here to allow gradient flow for lower uncertainties
        sigma = F.softplus(raw_sigma) + self.epsilon

        # Calculate absolute error
        abs_error = torch.abs(target - fvc_pred)

        # Compute the Negative Log Likelihood terms
        # Term 1: ln(sqrt(2) * sigma)
        term1 = torch.log(self.metric_constant * sigma)

        # Term 2: (sqrt(2) * abs_error) / sigma
        term2 = (self.metric_constant * abs_error) / sigma

        # Sum terms to get NLL
        loss = term1 + term2

        return torch.mean(loss)
