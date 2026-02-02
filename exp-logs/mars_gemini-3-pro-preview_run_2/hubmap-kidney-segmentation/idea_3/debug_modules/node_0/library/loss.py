import torch
import torch.nn as nn
from library.config import Config


class BCEDiceLoss(nn.Module):
    """
    Hybrid loss function combining Binary Cross Entropy (BCE) and Dice Loss.
    Useful for segmentation tasks with class imbalance, such as glomerulus detection.
    """

    def __init__(
        self,
        bce_weight=Config.LOSS_BCE_WEIGHT,
        dice_weight=Config.LOSS_DICE_WEIGHT,
        smooth=1.0,
    ):
        """
        Args:
            bce_weight (float): Weight for the BCE component.
            dice_weight (float): Weight for the Dice component.
            smooth (float): Smoothing factor for Dice calculation to avoid division by zero.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        Calculates the weighted sum of BCE and Dice loss.

        Args:
            inputs (torch.Tensor): Model predictions (logits) of shape (N, C, H, W) or (N, H, W).
            targets (torch.Tensor): Ground truth masks of shape (N, C, H, W) or (N, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Flatten inputs and targets to ensure computation over the whole batch
        # This treats the batch as a single volume, which is often stable for sparse segmentation
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1).float()

        # 1. Binary Cross Entropy Loss
        # BCEWithLogitsLoss takes logits directly (numerically stable)
        bce_loss = self.bce_loss(inputs_flat, targets_flat)

        # 2. Dice Loss
        # Apply sigmoid to convert logits to probabilities
        inputs_prob = torch.sigmoid(inputs_flat)

        intersection = (inputs_prob * targets_flat).sum()
        union = inputs_prob.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        # 3. Combined Loss
        total_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

        return total_loss
