import torch
import torch.nn as nn
from library.config import Config


class BatchDiceLoss(nn.Module):
    """
    Computes the Dice Loss over the entire batch (Batch-Level Dice).

    Instead of computing Dice for each sample and averaging, this treats
    the entire batch as a single volume. This aligns better with the
    Global Dice metric used in the competition and stabilizes gradients
    when masks are sparse or empty.
    """

    def __init__(self, smooth: float = 1e-6):
        super(BatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (before sigmoid), shape (B, C, H, W).
            targets: Binary ground truth masks, shape (B, C, H, W).

        Returns:
            Scalar tensor representing the Dice loss (1 - Dice Coefficient).
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors completely to treat the batch as one large volume
        # shape: (N,) where N = Batch * Channels * Height * Width
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate intersection and union over the whole batch
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Compute Dice Coefficient
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return Dice Loss
        return 1.0 - dice_score


class HybridLoss(nn.Module):
    """
    Combines Binary Cross Entropy (BCE) and Batch-Level Dice Loss.

    L_total = L_BCE + L_BatchDice

    BCE provides smooth, convex gradients for pixel-level classification,
    while Batch Dice directly optimizes the segmentation overlap metric.
    """

    def __init__(self):
        super(HybridLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = BatchDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output, shape (B, C, H, W).
            targets: Binary ground truth masks, shape (B, C, H, W).

        Returns:
            Total loss value.
        """
        # Ensure targets are float for BCE calculation
        if targets.dtype != logits.dtype:
            targets = targets.type_as(logits)

        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        return bce + dice
