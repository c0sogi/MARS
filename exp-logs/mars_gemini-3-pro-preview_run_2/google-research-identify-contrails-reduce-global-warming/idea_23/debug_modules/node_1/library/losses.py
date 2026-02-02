import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BatchDiceLoss(nn.Module):
    """
    Computes the Dice Loss over the entire flattened batch.

    This treats the batch as a single large volume, which helps stabilize gradients
    when masks are sparse and aligns with the global Dice metric used in evaluation.

    Args:
        smooth (float): Smoothing factor to avoid division by zero.
    """

    def __init__(self, smooth: float = 1e-6):
        super(BatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid) of shape (B, C, H, W).
            targets (torch.Tensor): Ground truth binary masks of shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value (1 - Dice).
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors to treat the batch as a single set of pixels
        # Shape becomes (N,) where N = B * C * H * W
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate intersection and union over the whole batch
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return Dice Loss
        return 1.0 - dice


class HybridLoss(nn.Module):
    """
    Composite loss function combining Binary Cross Entropy (BCE) and Batch Dice Loss.

    L_total = (bce_weight * L_BCE) + (dice_weight * L_BatchDice)

    Args:
        bce_weight (float): Weight for the BCE component. Defaults to Config.BCE_WEIGHT.
        dice_weight (float): Weight for the Dice component. Defaults to Config.DICE_WEIGHT.
    """

    def __init__(
        self,
        bce_weight: float = Config.BCE_WEIGHT,
        dice_weight: float = Config.DICE_WEIGHT,
    ):
        super(HybridLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        # Initialize components
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = BatchDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Raw model outputs.
            targets (torch.Tensor): Ground truth masks.

        Returns:
            torch.Tensor: Weighted combined loss.
        """
        # Compute BCE Loss
        bce = self.bce_loss(logits, targets)

        # Compute Batch Dice Loss
        dice = self.dice_loss(logits, targets)

        # Weighted sum
        total_loss = (self.bce_weight * bce) + (self.dice_weight * dice)

        return total_loss
