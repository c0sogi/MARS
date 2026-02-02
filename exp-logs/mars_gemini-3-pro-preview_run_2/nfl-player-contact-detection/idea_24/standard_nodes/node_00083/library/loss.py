import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import FOCAL_ALPHA, FOCAL_GAMMA


class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss for binary classification.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
        p_t is the model's estimated probability for the target class.
    """

    def __init__(
        self,
        alpha: float = FOCAL_ALPHA,
        gamma: float = FOCAL_GAMMA,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha (float): Weighting factor for the rare class (class 1).
                           Class 0 will be weighted by (1 - alpha).
            gamma (float): Focusing parameter to down-weight easy examples.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Logits from the model of shape (N, *) or (N, 1).
            targets (torch.Tensor): Ground truth labels of shape (N, *) or (N, 1).
                                    Should be 0 or 1 (float or int).

        Returns:
            torch.Tensor: The computed loss.
        """
        # Ensure targets are float for BCE calculation
        if targets.dtype != inputs.dtype:
            targets = targets.type_as(inputs)

        # Compute standard BCE with logits (no reduction yet to apply weights element-wise)
        # BCE_loss = - log(p_t)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # p_t = exp(-BCE_loss)
        pt = torch.exp(-bce_loss)

        # Calculate alpha_t
        # If target=1, alpha_t = alpha
        # If target=0, alpha_t = 1 - alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate Focal Loss
        # F_loss = alpha_t * (1 - p_t)^gamma * BCE_loss
        focal_loss = alpha_t * (1 - pt).pow(self.gamma) * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
