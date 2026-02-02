import torch
import torch.nn as nn
from library import config


class BatchDiceLoss(nn.Module):
    """
    Calculates the Dice Loss by treating the entire batch as a single volume.
    This helps stabilize gradients and aligns with the global Dice metric.
    """

    def __init__(self, smooth=1.0):
        super(BatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) Raw output from the model (before Sigmoid).
            targets: (B, C, H, W) or (B, H, W) Binary ground truth masks.
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Ensure targets match probs shape (B, C, H, W)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # Flatten the batch to treat it as a global volume
        # shape: (N,) where N = B * C * H * W
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate intersection and union
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Calculate Dice coefficient
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return Dice Loss
        return 1.0 - dice_score


class HybridLoss(nn.Module):
    """
    Combines Binary Cross Entropy (BCE) and Batch Dice Loss.
    BCE provides smooth gradients for pixel-level classification, while
    Dice Loss optimizes for the overlap metric directly.
    """

    def __init__(self):
        super(HybridLoss, self).__init__()
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.dice_loss_fn = BatchDiceLoss()

        self.bce_weight = config.LOSS_BCE_WEIGHT
        self.dice_weight = config.LOSS_DICE_WEIGHT

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) Raw output from the model.
            targets: (B, C, H, W) or (B, H, W) Binary ground truth masks.
        """
        # Ensure targets are float for BCE
        targets = targets.float()

        # If targets are (B, H, W), add channel dim for consistency with logits (B, 1, H, W)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # Calculate individual losses
        bce_loss = self.bce_loss_fn(logits, targets)
        dice_loss = self.dice_loss_fn(logits, targets)

        # Combine losses
        total_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

        return total_loss
