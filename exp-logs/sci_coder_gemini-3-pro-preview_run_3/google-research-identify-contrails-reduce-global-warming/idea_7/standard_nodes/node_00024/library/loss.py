import torch
import torch.nn as nn
from library.config import Config
from library.utils import dice_coef


class ContrailLoss(nn.Module):
    """
    A custom loss function for Contrail Identification that combines
    Binary Cross Entropy (BCE) Loss and Dice Loss.

    This hybrid loss helps address class imbalance (via Dice) while providing
    smooth gradients for pixel-level classification (via BCE).
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=Config.SMOOTH):
        """
        Args:
            bce_weight (float): Weight for the BCE component of the loss.
            dice_weight (float): Weight for the Dice component of the loss.
            smooth (float): Smoothing factor for Dice calculation to avoid division by zero.
        """
        super(ContrailLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        # BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class.
        # This is more numerically stable than using a plain Sigmoid followed by a BCELoss.
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Computes the combined loss.

        Args:
            logits (torch.Tensor): Raw output from the model (before sigmoid).
                                   Shape: (Batch, Channels, Height, Width)
            targets (torch.Tensor): Ground truth binary masks.
                                    Shape: (Batch, Channels, Height, Width)

        Returns:
            torch.Tensor: The calculated weighted loss.
        """
        # Ensure targets are float for BCE calculation
        targets = targets.float()

        # 1. Compute Binary Cross Entropy Loss
        # logits are raw scores, so we use BCEWithLogitsLoss
        bce_loss = self.bce(logits, targets)

        # 2. Compute Dice Loss
        # We need probabilities for Dice calculation, so we apply sigmoid to logits
        probs = torch.sigmoid(logits)

        # Calculate Dice coefficient
        # We use the provided utility function which flattens the input,
        # effectively computing the Global Dice over the batch.
        dice_score = dice_coef(probs, targets, smooth=self.smooth)

        # Dice Loss is 1 - Dice Coefficient
        dice_loss = 1.0 - dice_score

        # 3. Combine losses
        total_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

        return total_loss
