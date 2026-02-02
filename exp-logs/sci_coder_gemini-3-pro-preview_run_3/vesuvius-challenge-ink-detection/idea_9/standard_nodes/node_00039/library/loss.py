import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """
    Balanced Loss function combining Binary Cross Entropy (BCE) and Dice Loss.

    BCE addresses pixel-level classification accuracy, while Dice Loss optimizes
    for the geometric overlap (Intersection over Union) of the predicted and
    ground truth masks. This combination helps handle class imbalance and
    improves segmentation quality.
    """

    def __init__(self, bce_weight=0.5, smooth=1e-6):
        """
        Args:
            bce_weight (float): Weight assigned to the BCE component.
                                The Dice component will receive (1 - bce_weight).
            smooth (float): Smoothing factor to avoid division by zero in Dice calculation.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = 1.0 - bce_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model output (before sigmoid), shape (N, C, H, W) or (N, H, W).
            targets (torch.Tensor): Ground truth binary masks, same shape as logits.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. BCE Loss (operates on logits)
        bce = self.bce_loss(logits, targets)

        # 2. Dice Loss (operates on probabilities)
        probs = torch.sigmoid(logits)

        # Flatten tensors to compute intersection and union over the batch/image
        # Using view(-1) ensures we treat the entire batch as a continuous stream of pixels
        # or we can keep batch dim. Here we flatten entirely for a global Dice
        # which is often more stable for small batches.
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        denominator = probs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice_score

        # 3. Combined Loss
        total_loss = (self.bce_weight * bce) + (self.dice_weight * dice_loss)

        return total_loss
