import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the modified Laplace Log Likelihood loss for the Lung Function Decline prediction task.

    The loss is designed to optimize the competition metric:
    Metric = - (sqrt(2) * Delta) / Sigma_clipped - ln(sqrt(2) * Sigma_clipped)

    Therefore, we minimize the negative of the metric (ignoring constants and clipping for training stability):
    Loss = (sqrt(2) * |y_true - y_pred|) / sigma + ln(sigma)
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Args:
            eps (float): A small value to ensure numerical stability (avoid div by zero or log(0)).
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the Laplace Log Likelihood Loss.

        Args:
            preds (torch.Tensor): Model predictions of shape (batch_size, 2).
                                  - Column 0: Predicted FVC (y_pred)
                                  - Column 1: Predicted Confidence (sigma)
            targets (torch.Tensor): True FVC values of shape (batch_size, ) or (batch_size, 1).

        Returns:
            torch.Tensor: The calculated loss value.
        """
        # Ensure targets are the correct shape (batch_size, 1)
        if targets.dim() == 1:
            targets = targets.view(-1, 1)

        # Separate the predictions
        fvc_pred = preds[:, 0].view(-1, 1)
        sigma_pred = preds[:, 1].view(-1, 1)

        # Enforce positivity for sigma and apply Safe Floor.
        # Cite {solution_lesson_node_00013}: Prevent NLL Singularity by bounding uncertainty.
        # We use softplus to ensure smooth positivity and add a floor corresponding
        # to the metric's clip value (70ml) in the scaled space.
        sigma_min = Config.SIGMA_CLIP / Config.TARGET_STD
        sigma = F.softplus(sigma_pred) + sigma_min

        # Calculate the absolute error (Delta)
        # Note: We do NOT clip the error to 1000 during training to allow the model
        # to learn from large errors.
        delta = torch.abs(targets - fvc_pred)

        # Calculate the Loss
        # Formula: (sqrt(2) * delta) / sigma + ln(sigma)
        # We compute sqrt(2) on the correct device
        sqrt_2 = torch.tensor(2.0, device=preds.device).sqrt()

        loss = (sqrt_2 * delta) / sigma + torch.log(sigma)

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
