import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Loss function combining Binary Cross Entropy and Dice Loss.

    This hybrid loss is particularly effective for image segmentation tasks.
    BCE provides smooth gradients for pixel-wise classification, while Dice Loss
    directly optimizes the overlap metric (IoU/Dice) and handles class imbalance well.
    """

    def __init__(self, smooth=1.0):
        """
        Args:
            smooth (float): Smoothing factor to prevent division by zero in Dice calculation.
                            Default is 1.0.
        """
        super(BCEDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Calculate the combined BCE and Dice loss.

        Args:
            logits (torch.Tensor): Raw model predictions (before sigmoid) of shape (B, C, H, W) or (B, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks of shape (B, C, H, W) or (B, 1, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Ensure targets are float for numerical stability in BCE and multiplication
        targets = targets.float()

        # 1. Binary Cross Entropy Loss
        # F.binary_cross_entropy_with_logits applies sigmoid internally, which is more stable
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

        # 2. Dice Loss
        # Apply sigmoid to convert logits to probabilities [0, 1]
        probs = torch.sigmoid(logits)

        # Flatten the tensors to (Batch_Size, -1) to calculate Dice score per image
        # This preserves the batch dimension so we average the Dice score across the batch
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        # Calculate intersection and union
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        # Calculate Dice coefficient
        # Formula: 2*|A n B| / (|A| + |B|)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice Loss is 1 - Dice Score
        dice_loss = 1.0 - dice_score.mean()

        # Combine losses
        return bce_loss + dice_loss
