import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary classification.
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    This implementation wraps BCEWithLogitsLoss for numerical stability.
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Args:
            alpha (float): Weighting factor for the rare class (default: 0.25).
            gamma (float): Focusing parameter to down-weight easy examples (default: 2.0).
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid). Shape: [batch_size] or [batch_size, 1].
            targets (torch.Tensor): Ground truth binary labels (0 or 1). Shape: matches logits.

        Returns:
            torch.Tensor: Scalar loss value (mean over batch).
        """
        # Ensure targets are float for BCE calculation
        if targets.dtype != logits.dtype:
            targets = targets.type_as(logits)

        # Calculate Binary Cross Entropy with Logits
        # reduction='none' is required to apply the focal weights per sample
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # pt is the probability of the true class
        # bce_loss = -log(pt), so pt = exp(-bce_loss)
        pt = torch.exp(-bce_loss)

        # Calculate Focal Loss
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return loss.mean()
