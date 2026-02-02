import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Composite loss function combining Binary Cross Entropy (BCE) and Dice Loss.

    This loss is designed to optimize both pixel-level classification accuracy (via BCE)
    and geometric overlap (via Dice), which is crucial for the ink detection task
    where the target signal is sparse and the evaluation metric is overlap-based.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        """
        Initialize the BCEDiceLoss.

        Args:
            bce_weight (float): Weight for the BCE component.
            dice_weight (float): Weight for the Dice component.
            smooth (float): Smoothing factor to prevent division by zero in Dice calculation.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
        self.bce_fn = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        Forward pass to calculate the composite loss.

        Args:
            inputs (torch.Tensor): Raw model outputs (logits) of shape (Batch, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks of shape (Batch, 1, H, W).

        Returns:
            torch.Tensor: The calculated scalar loss.
        """
        # 1. Calculate BCE Loss
        # inputs are logits, targets are 0 or 1
        bce_loss = self.bce_fn(inputs, targets)

        # 2. Calculate Dice Loss
        # Apply sigmoid to convert logits to probabilities [0, 1]
        inputs_prob = torch.sigmoid(inputs)

        # Flatten the tensors to treat all pixels in the batch as a single vector
        # This calculates a global Dice score over the batch
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()

        # Dice coefficient: (2 * |A n B|) / (|A| + |B|)
        dice_score = (2.0 * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth
        )

        # Dice Loss is 1 - Dice Score
        dice_loss = 1.0 - dice_score

        # 3. Combine Losses
        total_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

        return total_loss
