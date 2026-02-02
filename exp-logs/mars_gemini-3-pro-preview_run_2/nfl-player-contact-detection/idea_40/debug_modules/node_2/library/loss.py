import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FocalLoss(nn.Module):
    """
    Implements Focal Loss for binary classification tasks.

    Focal Loss is designed to address class imbalance by down-weighting well-classified examples
    (easy negatives) and focusing training on hard negatives and positive examples.

    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha (float): Weighting factor for the positive class (0 < alpha < 1).
                       The negative class will be weighted by (1 - alpha).
                       Defaults to Config.FOCAL_ALPHA.
        gamma (float): Focusing parameter. Higher values down-weight easy examples more.
                       Defaults to Config.FOCAL_GAMMA.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                         Defaults to 'mean'.
    """

    def __init__(
        self,
        alpha: float = Config.FOCAL_ALPHA,
        gamma: float = Config.FOCAL_GAMMA,
        reduction: str = "mean",
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the focal loss.

        Args:
            inputs (torch.Tensor): Logits from the model (before sigmoid). Shape: (N, *)
            targets (torch.Tensor): Ground truth binary labels. Shape: (N, *)

        Returns:
            torch.Tensor: Computed loss.
        """
        # Ensure targets are float for BCE calculation
        if targets.dtype != inputs.dtype:
            targets = targets.type_as(inputs)

        # Compute binary cross entropy loss (log(p_t))
        # reduction='none' is required to apply focal weights element-wise
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate p_t (probability of the true class)
        # Since BCE = -log(p_t), p_t = exp(-BCE)
        pt = torch.exp(-bce_loss)

        # Calculate the focal term: (1 - p_t)^gamma
        focal_term = (1 - pt) ** self.gamma

        # Calculate the alpha weighting term
        # alpha_t = alpha if target=1 else (1-alpha)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * focal_term * bce_loss
        else:
            loss = focal_term * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
