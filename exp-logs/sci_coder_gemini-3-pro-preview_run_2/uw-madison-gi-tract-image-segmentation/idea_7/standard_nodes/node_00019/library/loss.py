import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss for segmentation tasks.

    This loss function combines the pixel-wise accuracy of BCE with the
    overlap-optimization of Dice Loss. It is particularly effective for
    segmentation tasks with class imbalance.
    """

    def __init__(self, smooth=1e-5):
        """
        Args:
            smooth (float): Smoothing factor to avoid division by zero in Dice calculation.
        """
        super(BCEDiceLoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Forward pass to calculate the combined loss.

        Args:
            logits (torch.Tensor): Raw model predictions (before sigmoid) of shape (B, C, H, W).
            targets (torch.Tensor): Ground truth binary masks of shape (B, C, H, W).

        Returns:
            torch.Tensor: The calculated scalar loss.
        """
        # BCE Loss
        # Ensure targets are float for BCEWithLogitsLoss
        bce_loss = self.bce(logits, targets.float())

        # Dice Loss
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W)
        # This allows calculating Dice per sample and per class
        batch_size = logits.size(0)
        num_classes = logits.size(1)

        probs_flat = probs.view(batch_size, num_classes, -1)
        targets_flat = targets.view(batch_size, num_classes, -1)

        # Calculate intersection and union
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)

        # Calculate Dice score
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice Loss = 1 - Dice Score
        # We average the dice loss across the batch and classes
        dice_loss = 1.0 - dice_score.mean()

        # Combine losses
        return bce_loss + dice_loss
