import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """
    Balanced Loss Function combining Binary Cross Entropy and Dice Loss.

    This loss is designed to optimize for segmentation overlap (Dice) while
    maintaining pixel-wise classification accuracy (BCE). It uses equal weighting
    to avoid skewing the probability landscape, relying on post-processing
    threshold tuning to adapt to the F0.5 metric.
    """

    def __init__(self, smooth=1e-6):
        """
        Args:
            smooth (float): Smoothing factor to avoid division by zero in Dice calculation.
        """
        super(BCEDiceLoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model predictions (logits) of shape (B, 1, H, W) or (B, H, W).
            targets (torch.Tensor): Ground truth labels (0 or 1) of same shape as inputs.

        Returns:
            torch.Tensor: The combined loss value.
        """
        # Calculate Binary Cross Entropy Loss (takes logits)
        # Ensure targets are float for BCE
        bce_loss = self.bce(inputs, targets.float())

        # Calculate Dice Loss
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(inputs)

        # Flatten the tensors to compute the metric over the entire batch or image
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()

        # Soft Dice Coefficient
        dice_score = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )

        dice_loss = 1.0 - dice_score

        # Combine losses with equal weighting
        return bce_loss + dice_loss
