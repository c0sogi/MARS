import torch
import torch.nn as nn
from library.config import Config


class HybridBatchDiceLoss(nn.Module):
    """
    A hybrid loss function combining Binary Cross Entropy (BCE) and Batch-level Dice Loss.

    Strategy:
    - BCE provides smooth, convex gradients for pixel-level classification stability.
    - Batch-level Dice treats the entire batch as a single volume (flattening N, H, W),
      which aligns the optimization with the Global Dice metric and handles empty masks
      more robustly than sample-averaged Dice.
    """

    def __init__(
        self, bce_weight=Config.BCE_WEIGHT, dice_weight=Config.DICE_WEIGHT, smooth=1.0
    ):
        """
        Initialize the HybridBatchDiceLoss.

        Args:
            bce_weight (float): Weight for the BCE component. Defaults to Config.BCE_WEIGHT.
            dice_weight (float): Weight for the Dice component. Defaults to Config.DICE_WEIGHT.
            smooth (float): Smoothing factor for Dice calculation to prevent division by zero.
        """
        super(HybridBatchDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Calculate the combined loss.

        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid) of shape (N, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks of shape (N, 1, H, W).

        Returns:
            torch.Tensor: The calculated scalar loss.
        """
        # 1. Binary Cross Entropy Component
        # BCEWithLogitsLoss handles the sigmoid internally for numerical stability
        bce = self.bce_loss(logits, targets)

        # 2. Batch-Level Dice Component
        # Apply sigmoid to get probabilities [0, 1]
        probs = torch.sigmoid(logits)

        # Flatten the tensors: (N, 1, H, W) -> (N * 1 * H * W)
        # This treats the whole batch as one large volume, aligning with Global Dice
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Dice coefficient formula: 2 * |X n Y| / (|X| + |Y|)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice Loss is 1 - Dice Coefficient
        dice_loss = 1.0 - dice_score

        # 3. Combine Losses
        total_loss = (self.bce_weight * bce) + (self.dice_weight * dice_loss)

        return total_loss
