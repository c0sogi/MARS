import torch
import torch.nn as nn
import numpy as np


class StandardizedLaplaceLoss(nn.Module):
    """
    Implements the Standardized Laplace Log Likelihood Loss.

    This loss function is designed to optimize the model in the standardized target space,
    aligning with the competition metric which is a modified Laplace Log Likelihood.

    Formula:
        L = (sqrt(2) * |y_scaled - mu_scaled|) / sigma_scaled + ln(sqrt(2) * sigma_scaled)
    """

    def __init__(self):
        super(StandardizedLaplaceLoss, self).__init__()
        # Pre-compute sqrt(2) as a constant
        self.sqrt_2 = np.sqrt(2)

    def forward(self, preds, targets):
        """
        Calculates the loss.

        Args:
            preds (torch.Tensor): Predictions from the model with shape (Batch_Size, 2).
                                  - Column 0: Predicted Mean (mu_scaled)
                                  - Column 1: Predicted Confidence (sigma_scaled).
                                    Assumed to be positive and floored by the model architecture.
            targets (torch.Tensor): Ground truth targets with shape (Batch_Size, 1) or (Batch_Size,).
                                    These should be standardized values.

        Returns:
            torch.Tensor: The scalar mean loss over the batch.
        """
        # Ensure targets have the shape (Batch_Size, 1) for broadcasting
        if targets.dim() == 1:
            targets = targets.view(-1, 1)

        # Unpack predictions
        # mu: Predicted FVC (scaled)
        # sigma: Predicted Confidence (scaled)
        mu = preds[:, 0:1]
        sigma = preds[:, 1:2]

        # Calculate absolute error (Delta)
        delta = torch.abs(targets - mu)

        # Calculate the two terms of the Laplace Negative Log Likelihood
        # Term 1: (sqrt(2) * Delta) / sigma
        term1 = (self.sqrt_2 * delta) / sigma

        # Term 2: ln(sqrt(2) * sigma)
        # We use torch.log for the natural logarithm
        term2 = torch.log(self.sqrt_2 * sigma)

        # Sum terms to get the loss per sample
        loss = term1 + term2

        # Return the mean loss over the batch
        return loss.mean()
