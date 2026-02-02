import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Composite loss function combining Binary Cross Entropy (BCE) and Soft Dice Loss.

    This loss is designed to handle the class imbalance in ink detection tasks.
    BCE provides smooth gradients for pixel-wise classification, while Dice Loss
    optimizes the intersection-over-union metric directly.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6, batch_dice=False):
        """
        Args:
            bce_weight (float): Weight for the BCE component.
            dice_weight (float): Weight for the Dice component.
            smooth (float): Smoothing factor to prevent division by zero in Dice calculation.
            batch_dice (bool): If True, computes Dice score globally across the entire batch.
                               If False, computes Dice per sample and averages them.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.batch_dice = batch_dice
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid), shape (B, C, H, W).
            targets (torch.Tensor): Ground truth binary masks, shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Binary Cross Entropy Loss
        # BCEWithLogitsLoss applies sigmoid internally for stability
        bce = self.bce_loss(logits, targets)

        # 2. Soft Dice Loss
        # Apply sigmoid to get probabilities [0, 1]
        probs = torch.sigmoid(logits)

        if self.batch_dice:
            # Flatten entire batch: (N)
            probs_flat = probs.view(-1)
            targets_flat = targets.view(-1)

            intersection = (probs_flat * targets_flat).sum()
            union = probs_flat.sum() + targets_flat.sum()

            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        else:
            # Flatten per sample: (B, N)
            # Assuming channel dim is 1, we flatten (C, H, W)
            batch_size = probs.shape[0]
            probs_flat = probs.view(batch_size, -1)
            targets_flat = targets.view(batch_size, -1)

            intersection = (probs_flat * targets_flat).sum(dim=1)
            union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_score = dice_score.mean()

        dice_loss = 1.0 - dice_score

        # 3. Composite Loss
        total_loss = (self.bce_weight * bce) + (self.dice_weight * dice_loss)

        return total_loss
