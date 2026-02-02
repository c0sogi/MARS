import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FocalLoss(nn.Module):
    """
    Implements Focal Loss for binary classification to address class imbalance.

    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    This implementation wraps BCEWithLogitsLoss for numerical stability.
    """

    def __init__(self, alpha=None, gamma=None, reduction="mean"):
        """
        Args:
            alpha (float, optional): Weighting factor for the positive class (0 < alpha < 1).
                                     Defaults to Config.ALPHA.
            gamma (float, optional): Focusing parameter (gamma >= 0).
                                     Defaults to Config.GAMMA.
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Defaults to 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha if alpha is not None else Config.ALPHA
        self.gamma = gamma if gamma is not None else Config.GAMMA
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits from the model output. Shape [N, *].
            targets (torch.Tensor): Ground truth labels (0 or 1). Shape must match inputs.

        Returns:
            torch.Tensor: The computed loss.
        """
        # Ensure targets are float and same shape/device as inputs
        if targets.dtype != inputs.dtype:
            targets = targets.to(inputs.dtype)

        if targets.shape != inputs.shape:
            targets = targets.view_as(inputs)

        # Compute binary cross entropy with logits
        # reduction='none' preserves the element-wise loss for weighting
        # BCE = -log(p_t)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate p_t (probability of the true class)
        # Since BCE = -log(p_t), then p_t = exp(-BCE)
        pt = torch.exp(-bce_loss)

        # Calculate the focal term: (1 - p_t)^gamma
        focal_term = (1.0 - pt).pow(self.gamma)

        # Calculate the alpha term
        # alpha_t = alpha if target=1 else (1 - alpha)
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
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
