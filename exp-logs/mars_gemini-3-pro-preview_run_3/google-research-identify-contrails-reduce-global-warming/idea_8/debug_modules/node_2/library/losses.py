import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DiceLoss(nn.Module):
    """
    Differentiable Dice Loss for binary segmentation.

    The Dice coefficient is defined as 2 * |X n Y| / (|X| + |Y|).
    The loss is defined as 1 - Dice.
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid), shape (B, C, H, W) or (B, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks, shape (B, C, H, W) or (B, 1, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors to compute Dice over the batch or image
        # Flattening helps in handling the global nature of the metric if desired,
        # or simply treating pixels as a set.
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """
    Hybrid loss combining Binary Cross Entropy (BCE) and Dice Loss.

    Loss = bce_weight * BCE + dice_weight * Dice
    """

    def __init__(
        self, bce_weight=Config.BCE_WEIGHT, dice_weight=Config.DICE_WEIGHT, smooth=1e-6
    ):
        super(DiceBCELoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs.
            targets (torch.Tensor): Ground truth masks.

        Returns:
            torch.Tensor: Weighted combined loss.
        """
        # Ensure targets are float for BCE and Dice calculations
        targets = targets.float()

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        total_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

        return total_loss
