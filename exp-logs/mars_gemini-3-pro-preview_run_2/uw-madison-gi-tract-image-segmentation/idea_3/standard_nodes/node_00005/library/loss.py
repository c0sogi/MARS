import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Implements a combined Binary Cross Entropy (BCE) and Dice Loss.

    This loss function combines the pixel-wise classification accuracy of BCE
    with the overlap metric of Dice Loss, which is particularly effective for
    segmentation tasks with class imbalance.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        """
        Args:
            bce_weight (float): Weighting factor for the BCE loss component.
            dice_weight (float): Weighting factor for the Dice loss component.
            smooth (float): Smoothing factor to prevent division by zero in Dice calculation.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        # BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class.
        # This is more numerically stable than using a plain Sigmoid followed by a BCELoss.
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        Calculates the combined loss.

        Args:
            inputs (torch.Tensor): Model predictions (logits) of shape (Batch, Channels, Height, Width).
            targets (torch.Tensor): Ground truth binary masks of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: The calculated scalar loss.
        """
        # 1. Calculate Binary Cross Entropy Loss
        # inputs are logits, so we use BCEWithLogitsLoss
        bce = self.bce_loss(inputs, targets)

        # 2. Calculate Dice Loss
        # Apply sigmoid to convert logits to probabilities [0, 1]
        inputs_prob = torch.sigmoid(inputs)

        # Flatten the tensors to compute Dice per sample and per class
        # Shape becomes (Batch, Channels, Height * Width)
        inputs_flat = inputs_prob.flatten(2)
        targets_flat = targets.flatten(2)

        # Calculate intersection: |X ∩ Y|
        intersection = (inputs_flat * targets_flat).sum(dim=-1)

        # Calculate sums: |X| + |Y|
        # Note: We sum the probabilities and the boolean targets directly
        union = inputs_flat.sum(dim=-1) + targets_flat.sum(dim=-1)

        # Compute Dice Coefficient: (2 * |X ∩ Y|) / (|X| + |Y|)
        # Add smooth to numerator and denominator to avoid divide-by-zero
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice Loss = 1 - Dice Coefficient
        # We take the mean over the batch and channels
        dice_loss = 1.0 - dice_score.mean()

        # 3. Combine Losses
        total_loss = (self.bce_weight * bce) + (self.dice_weight * dice_loss)

        return total_loss
