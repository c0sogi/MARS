import torch
import torch.nn as nn


class LaplaceNLLLoss(nn.Module):
    """
    Custom Loss function for the Time-Conditioned Deep-Semantic Network.
    Implements the Modified Laplace Log Likelihood: L = |y - mu| / sigma + ln(sigma).

    This loss function optimizes the model to predict the FVC (mu) and the
    uncertainty (sigma) simultaneously. It encourages the model to be confident (low sigma)
    when the error is low, and less confident (high sigma) when the error is high.
    """

    def __init__(self, eps=1e-6):
        """
        Args:
            eps (float): Small epsilon value for numerical stability.
        """
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps

    def forward(self, mu, sigma, target):
        """
        Computes the Laplace NLL Loss.

        Args:
            mu (torch.Tensor): Predicted FVC values. Shape (Batch_Size,) or (Batch_Size, 1).
            sigma (torch.Tensor): Predicted Confidence values (must be positive). Shape (Batch_Size,) or (Batch_Size, 1).
            target (torch.Tensor): Ground truth FVC values. Shape (Batch_Size, 1) or (Batch_Size,).

        Returns:
            torch.Tensor: The calculated loss (scalar, mean over batch).
        """
        # Ensure target shape matches predictions by flattening
        # The DataLoader typically returns target as (Batch, 1), while model output might be (Batch,)
        target = target.view(-1)
        mu = mu.view(-1)
        sigma = sigma.view(-1)

        # Safety clamp for sigma to avoid division by zero or log(0)
        # Although the model architecture guarantees positive sigma via softplus,
        # this acts as an additional safety layer.
        sigma = torch.clamp(sigma, min=self.eps)

        # Calculate the absolute error term: |y - mu|
        abs_error = torch.abs(target - mu)

        # Calculate the loss: (|y - mu| / sigma) + log(sigma)
        # This formula is derived from the negative log likelihood of the Laplace distribution.
        loss = (abs_error / sigma) + torch.log(sigma)

        # Return the mean loss over the batch
        return torch.mean(loss)
