import torch
import torch.nn as nn
from library.config import Config


class BatchDiceLoss(nn.Module):
    """
    Computes the Dice Loss over the entire batch as a single volume.

    Formula:
        Dice = (2 * |X n Y|) / (|X| + |Y|)
        Loss = 1 - Dice

    This strategy stabilizes gradients for small objects and sparse masks by
    aggregating statistics across the batch, preventing division-by-zero or
    high variance issues commonly seen with per-sample averaging on empty masks.
    """

    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth (float): Smoothing factor to prevent division by zero.
        """
        super(BatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid) of shape (B, C, H, W).
            targets (torch.Tensor): Ground truth binary masks of shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar Dice loss.
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors: (B, C, H, W) -> (N,)
        # This treats the entire batch as one large sample
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate intersection and union
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Compute Dice coefficient
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice_score


class HybridLoss(nn.Module):
    """
    Combines Binary Cross Entropy (BCE) and Batch-level Dice Loss.

    BCE provides smooth gradients for pixel-level classification, while
    Dice Loss optimizes for the overlap of the specific class (contrails),
    handling the severe class imbalance.
    """

    def __init__(self, bce_weight: float = None, dice_weight: float = None):
        """
        Args:
            bce_weight (float, optional): Weight for BCE component. Defaults to Config.BCE_WEIGHT.
            dice_weight (float, optional): Weight for Dice component. Defaults to Config.DICE_WEIGHT.
        """
        super(HybridLoss, self).__init__()

        # Use config defaults if specific weights are not provided
        self.bce_weight = bce_weight if bce_weight is not None else Config.BCE_WEIGHT
        self.dice_weight = (
            dice_weight if dice_weight is not None else Config.DICE_WEIGHT
        )

        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.dice_loss_fn = BatchDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Raw model outputs.
            targets (torch.Tensor): Ground truth masks.

        Returns:
            torch.Tensor: Weighted sum of BCE and Dice losses.
        """
        bce = self.bce_loss_fn(logits, targets)
        dice = self.dice_loss_fn(logits, targets)

        total_loss = (self.bce_weight * bce) + (self.dice_weight * dice)

        return total_loss
