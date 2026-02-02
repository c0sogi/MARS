import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Hybrid loss function combining Binary Cross Entropy (BCE) and Dice Loss.

    This loss is effective for segmentation tasks with class imbalance, such as
    ink detection, where the foreground (ink) is sparse compared to the background.

    BCE handles pixel-wise classification accuracy, while Dice Loss optimizes
    the intersection-over-union (IoU) metric directly.
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
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        Forward pass.

        Args:
            inputs (torch.Tensor): Predicted logits from the model. Shape (B, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks. Shape (B, 1, H, W).

        Returns:
            torch.Tensor: The calculated weighted loss.
        """
        # 1. Binary Cross Entropy Loss
        # BCEWithLogitsLoss takes raw logits and applies Sigmoid internally for stability
        bce_loss = self.bce_loss_fn(inputs, targets)

        # 2. Dice Loss
        # Apply sigmoid to convert logits to probabilities [0, 1]
        probs = torch.sigmoid(inputs)

        # Flatten the tensors to compute the global Dice score for the batch
        # or compute per image and average. Here we flatten (B, 1, H, W) -> (N,)
        inputs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        union = inputs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        # 3. Combine Losses
        total_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

        return total_loss
