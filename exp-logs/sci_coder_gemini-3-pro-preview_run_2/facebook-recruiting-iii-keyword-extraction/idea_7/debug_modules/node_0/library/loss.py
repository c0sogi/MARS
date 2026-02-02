import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FocalLoss(nn.Module):
    """
    Focal Loss for dense multi-label classification.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
        p_t = p if y=1 else 1-p
        alpha_t = alpha if y=1 else 1-alpha

    Args:
        alpha (float): Weighting factor for the positive class (0 < alpha < 1).
                       If -1, no alpha weighting is applied.
        gamma (float): Focusing parameter to down-weight easy examples.
        reduction (str): Specifies the reduction to apply to the output:
                         'none' | 'mean' | 'sum'.
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
        Forward pass of the Focal Loss.

        Args:
            inputs (torch.Tensor): Logits from the model of shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels of shape (batch_size, num_classes).
                                    Should be binary (0 or 1).

        Returns:
            torch.Tensor: The calculated loss.
        """
        # Ensure targets are float for BCE calculation
        targets = targets.float()

        # Calculate standard Binary Cross Entropy loss
        # reduction='none' is required to apply the focal weights element-wise
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate probabilities
        # p = sigmoid(inputs)
        p = torch.sigmoid(inputs)

        # Calculate p_t (probability of the ground truth class)
        # If target=1, p_t = p
        # If target=0, p_t = 1 - p
        p_t = p * targets + (1 - p) * (1 - targets)

        # Calculate the focal modulating factor: (1 - p_t)^gamma
        focal_factor = (1 - p_t) ** self.gamma

        # Apply the focal factor to the BCE loss
        loss = focal_factor * bce_loss

        # Apply alpha weighting if specified
        if self.alpha >= 0:
            # alpha_t = alpha if target=1 else (1 - alpha)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
