import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    """
    Hybrid loss function combining Binary Cross Entropy (BCE) and Dice Loss.

    This loss is designed for binary segmentation tasks where the evaluation metric
    is the Dice coefficient. Combining BCE helps with pixel-wise convergence, while
    Dice Loss directly optimizes the overlap metric and handles class imbalance better.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        """
        Args:
            bce_weight (float): Weight for the BCE component of the loss.
            dice_weight (float): Weight for the Dice component of the loss.
            smooth (float): Smoothing factor to prevent division by zero in Dice calculation.
        """
        super(DiceBCELoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        # BCEWithLogitsLoss is numerically stable as it combines Sigmoid and BCE
        self.bce_fn = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        Computes the combined loss.

        Args:
            inputs (torch.Tensor): Model predictions (logits) of shape (B, C, H, W) or (B, H, W).
            targets (torch.Tensor): Ground truth binary masks of the same shape as inputs.

        Returns:
            torch.Tensor: The calculated scalar loss.
        """
        # 1. Binary Cross Entropy Loss
        # inputs are logits, so we use BCEWithLogitsLoss
        bce_loss = self.bce_fn(inputs, targets)

        # 2. Dice Loss
        # Apply sigmoid to convert logits to probabilities for Dice calculation
        inputs_prob = torch.sigmoid(inputs)

        # Flatten the tensors to treat the batch as a continuous stream of pixels
        # This calculates a "Global Dice" over the batch, which aligns well with the
        # competition metric (Global Dice over the test set).
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        union = inputs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        # 3. Combined Loss
        total_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

        return total_loss
