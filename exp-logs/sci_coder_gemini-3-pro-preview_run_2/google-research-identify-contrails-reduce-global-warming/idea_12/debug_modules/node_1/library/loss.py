import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BatchDiceLoss(nn.Module):
    """
    Calculates the Dice Loss over the entire batch volume.

    Instead of averaging the Dice score per image, this computes the intersection
    and union over the flattened batch (B, C, H, W). This aligns the loss function
    more closely with the global Dice metric used for evaluation and stabilizes
    training by treating the batch as a large sample.
    """

    def __init__(self, smooth=1e-6):
        """
        Args:
            smooth (float): Smoothing factor to prevent division by zero.
        """
        super(BatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid) of shape (B, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks of shape (B, 1, H, W).

        Returns:
            torch.Tensor: Scalar Dice loss (1 - Dice Coefficient).
        """
        # Apply sigmoid to convert logits to probabilities [0, 1]
        probs = torch.sigmoid(logits)

        # Flatten the tensors to treat the entire batch as a single volume
        # Shape becomes (N,) where N = B * 1 * H * W
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate Intersection and Union over the batch
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Compute Dice Coefficient
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return Dice Loss
        return 1.0 - dice_score


class HybridLoss(nn.Module):
    """
    Combines Binary Cross Entropy (BCE) and Batch-Level Dice Loss.

    L_total = bce_weight * L_BCE + dice_weight * L_BatchDice

    BCE provides smooth, convex gradients for pixel-level classification, while
    BatchDiceLoss optimizes for the global overlap metric directly.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1e-6):
        """
        Args:
            bce_weight (float): Weight for the BCE component.
            dice_weight (float): Weight for the Dice component.
            smooth (float): Smoothing factor for Dice calculation.
        """
        super(HybridLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

        # Standard BCEWithLogitsLoss (combines Sigmoid + BCELoss for stability)
        self.bce_loss = nn.BCEWithLogitsLoss()

        # Custom Batch-Level Dice Loss
        self.dice_loss = BatchDiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model outputs of shape (B, 1, H, W).
            targets (torch.Tensor): Ground truth masks of shape (B, 1, H, W).

        Returns:
            torch.Tensor: Weighted combined loss.
        """
        # Ensure targets are float for BCE calculation
        targets = targets.float()

        # Calculate individual losses
        loss_bce = self.bce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)

        # Combine losses
        total_loss = (self.bce_weight * loss_bce) + (self.dice_weight * loss_dice)

        return total_loss
