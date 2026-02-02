import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchDiceLoss(nn.Module):
    """
    Computes the Dice Loss over the entire batch (global/volumetric Dice).

    Formula: 1 - (2 * |X n Y| + smooth) / (|X| + |Y| + smooth)

    Treating the batch as a single volume stabilizes gradients and aligns
    better with the global competition metric compared to averaging sample-wise Dice scores.
    """

    def __init__(self, smooth=1e-6):
        """
        Args:
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super(BatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid) of shape (B, C, H, W).
            targets (torch.Tensor): Binary ground truth masks of shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors to compute the metric over the entire input batch
        # This aggregates statistics across all samples in the batch
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate intersection and union
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Compute Dice coefficient
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return Dice Loss (1 - Dice Coefficient)
        return 1.0 - dice_score


class HybridBCEDiceLoss(nn.Module):
    """
    Hybrid loss function combining Binary Cross Entropy (BCE) and Batch-Level Dice Loss.

    L_total = L_BCE + L_BatchDice

    BCE provides smooth, convex gradients for pixel-level classification, while
    Batch Dice directly optimizes the global overlap metric.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1e-6):
        """
        Args:
            bce_weight (float): Weight for the BCE component.
            dice_weight (float): Weight for the Dice component.
            smooth (float): Smoothing factor for Dice calculation.
        """
        super(HybridBCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        # BCEWithLogitsLoss combines Sigmoid and BCE for numerical stability
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = BatchDiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs of shape (B, C, H, W).
            targets (torch.Tensor): Binary ground truth masks of shape (B, C, H, W).

        Returns:
            torch.Tensor: Weighted sum of BCE and Dice losses.
        """
        # Ensure targets are float for BCE calculation
        targets = targets.float()

        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        return (self.bce_weight * bce) + (self.dice_weight * dice)
