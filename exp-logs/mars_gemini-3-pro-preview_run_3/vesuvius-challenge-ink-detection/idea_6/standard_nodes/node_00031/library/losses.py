import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Implements a balanced loss function combining Binary Cross Entropy (BCE)
    and Soft Dice Loss.

    This combination helps the model converge by optimizing for both
    pixel-wise accuracy (BCE) and geometric overlap (Dice), addressing
    class imbalance without extreme weighting.
    """

    def __init__(self, epsilon=1e-7):
        """
        Args:
            epsilon (float): Small constant to avoid division by zero in Dice calculation.
        """
        super(BCEDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Computes the combined BCE + Dice loss.

        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid) of shape (B, C, H, W).
            targets (torch.Tensor): Ground truth binary masks of shape (B, C, H, W) or (B, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Ensure targets are float and have the correct shape
        if targets.dtype != logits.dtype:
            targets = targets.type_as(logits)

        if targets.dim() == logits.dim() - 1:
            targets = targets.unsqueeze(1)

        # 1. BCE Loss (Pixel-wise)
        bce_loss = self.bce(logits, targets)

        # 2. Soft Dice Loss (Geometric overlap)
        probs = torch.sigmoid(logits)

        # Flatten tensors for global dice computation
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
        dice_loss = 1.0 - dice_score

        # Combined Loss
        total_loss = bce_loss + dice_loss

        return total_loss
