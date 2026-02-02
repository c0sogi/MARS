import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridBCEDiceLoss(nn.Module):
    """
    Hybrid loss function combining Binary Cross Entropy (BCE) and Dice Loss.
    Designed to handle class imbalance in segmentation tasks by optimizing
    both pixel-wise classification and region overlap.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
        """
        Args:
            bce_weight (float): Weight for the BCE component.
            dice_weight (float): Weight for the Dice component.
            smooth (float): Smoothing factor for Dice calculation to avoid division by zero.
        """
        super(HybridBCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model predictions (logits) of shape (B, C, H, W) or (B, H, W).
            targets (torch.Tensor): Ground truth masks of shape (B, C, H, W) or (B, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Ensure inputs and targets have the same shape
        if inputs.shape != targets.shape:
            targets = targets.view_as(inputs)

        # Flatten inputs and targets to (N,)
        inputs_flat = inputs.reshape(-1)
        targets_flat = targets.reshape(-1)

        # 1. Binary Cross Entropy Loss
        # inputs are logits, so we use binary_cross_entropy_with_logits for numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(inputs_flat, targets_flat.float())

        # 2. Soft Dice Loss
        # Apply sigmoid to get probabilities
        inputs_soft = torch.sigmoid(inputs_flat)

        intersection = (inputs_soft * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            inputs_soft.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1.0 - dice_score

        # Combined Loss
        total_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

        return total_loss


class DiceScore(nn.Module):
    """
    Metric class to calculate the Hard Dice Coefficient (F1 Score).
    Used for monitoring model performance during validation.
    """

    def __init__(self, threshold=Config.MASK_THRESHOLD, smooth=1e-6):
        """
        Args:
            threshold (float): Threshold to convert probabilities to binary mask.
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super(DiceScore, self).__init__()
        self.threshold = threshold
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model predictions (logits).
            targets (torch.Tensor): Ground truth masks.

        Returns:
            torch.Tensor: Scalar Dice score.
        """
        # Apply sigmoid to convert logits to probabilities
        inputs = torch.sigmoid(inputs)

        # Ensure inputs and targets have the same shape
        if inputs.shape != targets.shape:
            targets = targets.view_as(inputs)

        # Flatten tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        # Binarize predictions based on threshold (Hard Dice)
        inputs = (inputs > self.threshold).float()

        # Calculate Dice
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )

        return dice
