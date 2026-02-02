import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss for Semantic Segmentation.

    This loss function combines:
    1. BCEWithLogitsLoss: Measures pixel-wise classification error.
    2. Dice Loss: Measures overlap between predicted and ground truth masks.

    The combination helps in stabilizing training and handling class imbalance.
    """

    def __init__(
        self, bce_weight=Config.BCE_WEIGHT, dice_weight=Config.DICE_WEIGHT, smooth=1e-6
    ):
        """
        Args:
            bce_weight (float): Weight for the BCE component.
            dice_weight (float): Weight for the Dice component.
            smooth (float): Smoothing factor to prevent division by zero in Dice calculation.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        """
        Forward pass.

        Args:
            y_pred (torch.Tensor): Predicted logits of shape (B, C, H, W).
            y_true (torch.Tensor): Ground truth masks of shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Ensure targets are float for BCE
        y_true = y_true.float()

        # 1. Binary Cross Entropy Loss
        bce = self.bce_loss(y_pred, y_true)

        # 2. Dice Loss
        # Apply sigmoid to convert logits to probabilities
        pred_probs = torch.sigmoid(y_pred)

        # Flatten spatial dimensions: (B, C, H, W) -> (B, C, N)
        # This allows computing Dice per sample and per class independently
        batch_size = y_true.size(0)
        num_classes = y_true.size(1)

        pred_flat = pred_probs.view(batch_size, num_classes, -1)
        true_flat = y_true.view(batch_size, num_classes, -1)

        # Intersection and Union
        intersection = (pred_flat * true_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + true_flat.sum(dim=2)

        # Dice coefficient per channel per sample
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice Loss = 1 - Mean Dice Score
        dice_loss = 1.0 - dice_score.mean()

        # Weighted Combination
        total_loss = (self.bce_weight * bce) + (self.dice_weight * dice_loss)

        return total_loss
