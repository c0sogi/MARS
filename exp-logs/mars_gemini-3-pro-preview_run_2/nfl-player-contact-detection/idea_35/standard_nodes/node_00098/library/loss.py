import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss implementation for addressing class imbalance.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
        p_t = p if y=1 else 1-p
        alpha_t = alpha if y=1 else 1-alpha
        gamma is the focusing parameter.

    This implementation wraps BCEWithLogitsLoss for numerical stability.
    """

    def __init__(
        self,
        alpha: float = Config.FOCAL_ALPHA,
        gamma: float = Config.FOCAL_GAMMA,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha (float): Weighting factor for the positive class (class 1).
                           The negative class (class 0) will be weighted by (1 - alpha).
                           Default: 0.25 (from Config).
            gamma (float): Focusing parameter to down-weight easy examples.
                           Default: 2.0 (from Config).
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss function.

        Args:
            logits (torch.Tensor): Predicted logits (before sigmoid) of shape (N, 1) or (N,).
            targets (torch.Tensor): Ground truth binary labels of shape (N, 1) or (N,).

        Returns:
            torch.Tensor: The computed loss.
        """
        # Ensure targets are float and match logits shape
        targets = targets.view_as(logits).float()

        # Compute standard BCE loss (element-wise)
        # using binary_cross_entropy_with_logits for numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Compute probabilities
        probs = torch.sigmoid(logits)

        # Calculate p_t: probability associated with the ground truth class
        # p_t = p if target=1, else 1-p
        p_t = torch.where(targets == 1, probs, 1 - probs)

        # Calculate alpha_t: weighting factor associated with the ground truth class
        # alpha_t = alpha if target=1, else 1-alpha
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Calculate the modulating factor: (1 - p_t)^gamma
        modulating_factor = torch.pow(1 - p_t, self.gamma)

        # Combine terms
        focal_loss = alpha_t * modulating_factor * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
