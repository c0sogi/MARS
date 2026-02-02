import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DiceLoss(nn.Module):
    """
    Differentiable Dice Loss for semantic segmentation.
    Computes the Dice coefficient between predicted probabilities and ground truth masks.
    """

    def __init__(self, smooth=1e-6):
        """
        Args:
            smooth (float): Smoothing factor to prevent division by zero and smooth the loss.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs of shape (B, C, H, W).
            targets (torch.Tensor): Ground truth masks of shape (B, C, H, W).
                                    Should be 0.0 or 1.0.

        Returns:
            torch.Tensor: Scalar loss value (1 - Dice).
        """
        # Apply sigmoid to convert logits to probabilities [0, 1]
        probs = torch.sigmoid(logits)

        # Flatten the tensors to (B, C, N) where N = H * W
        # This allows computing the metric per sample and per class
        batch_size = logits.size(0)
        num_classes = logits.size(1)

        probs_flat = probs.view(batch_size, num_classes, -1)
        targets_flat = targets.view(batch_size, num_classes, -1)

        # Calculate intersection and union
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)

        # Compute Dice coefficient
        # Shape: (B, C)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return 1 - mean dice score
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Composite loss function combining Binary Cross Entropy (BCE) and Dice Loss.
    Used to optimize both pixel-level classification accuracy and geometric overlap.
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()
        # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

        # Load weights from configuration
        self.ce_weight = Config.LOSS_CE_WEIGHT
        self.dice_weight = Config.LOSS_DICE_WEIGHT

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs of shape (B, C, H, W).
            targets (torch.Tensor): Ground truth masks of shape (B, C, H, W).

        Returns:
            torch.Tensor: Weighted combined loss.
        """
        # Ensure targets are float for BCE
        targets = targets.float()

        # Calculate individual losses
        loss_bce = self.bce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)

        # Combine
        return (self.ce_weight * loss_bce) + (self.dice_weight * loss_dice)
