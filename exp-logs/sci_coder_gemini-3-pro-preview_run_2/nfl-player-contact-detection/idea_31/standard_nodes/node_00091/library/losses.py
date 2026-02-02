import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FocalLoss(nn.Module):
    """
    Implements Focal Loss for binary classification.

    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    This implementation wraps BCEWithLogitsLoss for numerical stability.
    It addresses class imbalance by down-weighting well-classified examples (via gamma)
    and balancing positive/negative classes (via alpha).
    """

    def __init__(
        self, alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA, reduction="mean"
    ):
        """
        Args:
            alpha (float): Weighting factor for the positive class (class 1).
                           The negative class (class 0) will be weighted by (1 - alpha).
                           Default is Config.FOCAL_ALPHA (0.25).
            gamma (float): Focusing parameter. Higher values reduce the loss contribution
                           of easy examples. Default is Config.FOCAL_GAMMA (2.0).
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits from the model (before sigmoid). Shape: (N, *)
            targets (torch.Tensor): Ground truth binary labels. Shape: (N, *)

        Returns:
            torch.Tensor: The computed loss.
        """
        # Ensure targets are float for calculation
        targets = targets.float()

        # Reshape targets to match inputs if necessary (e.g. if inputs are (N,1) and targets are (N,))
        if targets.shape != inputs.shape:
            targets = targets.view_as(inputs)

        # Compute standard BCE loss (element-wise)
        # binary_cross_entropy_with_logits is numerically stable
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate p_t (probability of the true class)
        # Since BCE = -log(p_t), we can compute p_t = exp(-BCE)
        pt = torch.exp(-bce_loss)

        # Calculate alpha weighting
        # alpha_t = alpha if target=1 else (1-alpha)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate the modulating factor: (1 - p_t)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Combine terms
        loss = alpha_t * focal_weight * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
