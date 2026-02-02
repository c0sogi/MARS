import torch
import torch.nn as nn
from library.config import Config


class DiceBCELoss(nn.Module):
    """
    Hybrid loss function combining Binary Cross Entropy (BCE) and Dice Loss.

    BCEWithLogitsLoss is used for numerical stability.
    Dice Loss is computed on the sigmoid-activated probabilities to directly optimize
    the competition metric (Dice Coefficient).
    """

    def __init__(self, bce_weight=None):
        """
        Args:
            bce_weight (torch.Tensor, optional): A manual rescaling weight given to the loss of each batch element.
                                                 If given, has to be a Tensor of size nbatch.
        """
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(weight=bce_weight)
        self.smooth = Config.SMOOTH

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model predictions (logits) of shape (B, C, H, W) or (B, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks of the same shape.

        Returns:
            torch.Tensor: Scalar loss value (BCE + Dice Loss).
        """
        # 1. Binary Cross Entropy Loss
        # inputs are logits, targets are 0 or 1
        bce_loss = self.bce(inputs, targets)

        # 2. Dice Loss
        # Apply sigmoid to convert logits to probabilities
        inputs_prob = torch.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()

        # Soft Dice calculation
        dice_score = (2.0 * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1.0 - dice_score

        # Combine losses
        return bce_loss + dice_loss
