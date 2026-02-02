import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Soft Dice Loss.

    This loss function combines pixel-wise accuracy (BCE) with a metric that
    directly optimizes the overlap between predicted and ground truth masks (Dice).
    It is particularly useful for segmentation tasks with class imbalance.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
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

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model predictions (logits) of shape (N, C, H, W).
            targets (torch.Tensor): Ground truth masks of shape (N, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # BCE Loss (BCEWithLogitsLoss handles sigmoid internally for stability)
        bce = self.bce_loss(inputs, targets)

        # Dice Loss
        # Apply sigmoid to convert logits to probabilities
        inputs_prob = torch.sigmoid(inputs)

        # Flatten label and prediction tensors to compute Dice per sample/channel
        # Shape transformation: (N, C, H, W) -> (N * C, H * W)
        # This treats every channel in every sample as an independent binary segmentation task
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth
        )

        dice_loss = 1.0 - dice_score

        # Combined Loss
        total_loss = (self.bce_weight * bce) + (self.dice_weight * dice_loss)

        return total_loss
