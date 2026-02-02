import torch
import torch.nn as nn
import numpy as np
from library.config import Config


class MetricAlignedLaplaceLoss(nn.Module):
    """
    Implements the Metric-Aligned Laplace Log Likelihood Loss in the standardized space.

    Formula:
        L = (sqrt(2) * |y_scaled - mu_scaled| / sigma_scaled) + ln(sqrt(2) * sigma_scaled)

    This loss function corresponds to the negative log likelihood of a Laplace distribution,
    explicitly including the sqrt(2) constants to align the optimization landscape with
    the competition's evaluation metric.
    """

    def __init__(self):
        super(MetricAlignedLaplaceLoss, self).__init__()
        self.sqrt_2 = np.sqrt(2)

    def forward(self, preds, targets):
        """
        Calculates the loss.

        Args:
            preds (torch.Tensor): Model predictions of shape (batch_size, 2).
                                  - Column 0: mu_scaled (Predicted Mean)
                                  - Column 1: sigma_scaled (Predicted Confidence, must be > 0)
            targets (torch.Tensor): True targets of shape (batch_size, 1) or (batch_size).
                                    These should be standardized FVC values.

        Returns:
            torch.Tensor: The mean loss over the batch.
        """
        # Separate the predictions
        mu = preds[:, 0]
        sigma = preds[:, 1]

        # Ensure targets are flattened to match the dimension of mu and sigma
        y_true = targets.view(-1)

        # Calculate the absolute difference (Delta in standardized space)
        # Note: We do not clip the delta at 1000 here (as in the metric) because
        # we want the gradients to penalize large errors during training.
        abs_diff = torch.abs(y_true - mu)

        # Calculate the first term: (sqrt(2) * |y - mu|) / sigma
        term1 = (self.sqrt_2 * abs_diff) / sigma

        # Calculate the second term: ln(sqrt(2) * sigma)
        # Note: sigma is guaranteed to be positive by the model architecture (Softplus + epsilon)
        term2 = torch.log(self.sqrt_2 * sigma)

        # Sum the terms to get the negative log likelihood
        loss = term1 + term2

        # Return the mean loss over the batch
        return torch.mean(loss)
