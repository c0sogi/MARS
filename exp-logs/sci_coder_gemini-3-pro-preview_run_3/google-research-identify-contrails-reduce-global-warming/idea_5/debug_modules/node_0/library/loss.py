import torch
import torch.nn as nn
from library.utils import dice_coeff


class HybridLoss(nn.Module):
    """
    A hybrid loss function combining Binary Cross-Entropy (BCE) and Dice Loss.
    This is particularly effective for segmentation tasks with class imbalance,
    such as contrail detection.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        """
        Args:
            bce_weight (float): Weight for the BCE component.
            dice_weight (float): Weight for the Dice component.
            smooth (float): Smoothing factor for Dice calculation.
        """
        super(HybridLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        # BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class.
        # This is more numerically stable than using a plain Sigmoid followed by a BCELoss.
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        Computes the hybrid loss.

        Args:
            inputs (torch.Tensor): Model predictions (logits) of shape (B, C, H, W).
            targets (torch.Tensor): Ground truth masks of shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Compute Binary Cross-Entropy Loss
        # inputs are logits, targets should be float for BCEWithLogitsLoss
        bce = self.bce_loss(inputs, targets.float())

        # Compute Dice Loss
        # We need to apply sigmoid to logits to get probabilities for Dice calculation
        probs = torch.sigmoid(inputs)

        # Calculate Dice coefficient using the utility function
        # dice_coeff returns the Dice Score (higher is better)
        # We want to minimize loss, so we use (1 - Dice Score)
        dice_score = dice_coeff(probs, targets, self.smooth)
        dice = 1.0 - dice_score

        # Combine losses
        total_loss = (self.bce_weight * bce) + (self.dice_weight * dice)

        return total_loss
