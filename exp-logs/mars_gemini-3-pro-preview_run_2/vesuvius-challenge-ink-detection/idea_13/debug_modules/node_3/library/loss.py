import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """
    Composite loss function combining Binary Cross Entropy (BCE) and Dice Loss.
    Used to handle class imbalance and optimize segmentation overlap in the
    Siamese Multi-View SegFormer training pipeline.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-7):
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
        # BCEWithLogitsLoss is more numerically stable than Sigmoid + BCELoss
        self.bce_func = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model predictions (logits) of shape (B, 1, H, W).
            targets (torch.Tensor): Ground truth labels of shape (B, 1, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. BCE Loss (with Logits)
        bce_loss = self.bce_func(inputs, targets)

        # 2. Dice Loss
        # Apply sigmoid to convert logits to probabilities for Dice calculation
        probs = torch.sigmoid(inputs)

        # Flatten tensors to (N,)
        # Computing Dice over the entire batch treats it as one large volume.
        # This provides stability, especially when some images in the batch
        # contain no ink pixels (empty targets).
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Dice Coefficient: 2*|A n B| / (|A| + |B|)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        # 3. Composite Loss
        total_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

        return total_loss
