import torch
import torch.nn as nn
from library.config import Config


class BatchDiceLoss(nn.Module):
    """
    Calculates the Dice Loss over the entire batch volume.

    Instead of averaging the Dice score per sample, this loss treats the
    entire batch as a single volume (N * H * W). This aligns with the
    competition metric (Global Dice) and provides more stable gradients
    when masks are sparse or empty.
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
            logits (torch.Tensor): Raw model predictions (before sigmoid) of shape (B, C, H, W).
            targets (torch.Tensor): Ground truth binary masks of shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value (1 - Dice).
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors to treat the batch as a global set
        # Shape becomes (B * C * H * W,)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate intersection and union
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Compute Dice coefficient
        dice = (2.0 * intersection) / (union + self.smooth)

        # Return Dice Loss
        return 1.0 - dice


class HybridLoss(nn.Module):
    """
    Hybrid Loss combining Binary Cross Entropy (BCE) and Batch-Level Dice Loss.

    BCE provides smooth, convex gradients for pixel-level classification,
    while Dice Loss directly optimizes the overlap metric and handles class imbalance.
    """

    def __init__(self, bce_weight=Config.BCE_WEIGHT, dice_weight=Config.DICE_WEIGHT):
        """
        Args:
            bce_weight (float): Weight for the BCE component. Defaults to Config.BCE_WEIGHT.
            dice_weight (float): Weight for the Dice component. Defaults to Config.DICE_WEIGHT.
        """
        super(HybridLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        # BCEWithLogitsLoss includes Sigmoid internally for numerical stability
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = BatchDiceLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model predictions (before sigmoid).
            targets (torch.Tensor): Ground truth binary masks.

        Returns:
            torch.Tensor: Weighted sum of BCE and Dice loss.
        """
        # Calculate components
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        # Weighted sum
        total_loss = (self.bce_weight * bce) + (self.dice_weight * dice)

        return total_loss
