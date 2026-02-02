import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy (BCE) and Soft Dice Loss for 3D Segmentation.

    This loss function helps to mitigate class imbalance issues common in medical
    image segmentation by combining the pixel-wise accuracy of BCE with the
    overlap-based metric of Dice.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        """
        Args:
            bce_weight (float): Weight for the Binary Cross Entropy component.
            dice_weight (float): Weight for the Dice Loss component.
            smooth (float): Smoothing factor to prevent division by zero in Dice calculation.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Computes the combined loss.

        Args:
            logits (torch.Tensor): Model output (raw logits) of shape (B, C, D, H, W).
            targets (torch.Tensor): Ground truth binary masks of shape (B, C, D, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # --- Binary Cross Entropy Loss ---
        # BCEWithLogitsLoss is numerically more stable than Sigmoid + BCELoss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

        # --- Soft Dice Loss ---
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors to (B * C, -1) to compute Dice per channel/sample or globally.
        # Here we compute it per batch/channel and average.
        # Inputs: (B, C, D, H, W) -> (B, C, N) where N = D*H*W
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        probs_flat = probs.view(batch_size, num_classes, -1)
        targets_flat = targets.view(batch_size, num_classes, -1)

        # Intersection: (B, C)
        intersection = (probs_flat * targets_flat).sum(dim=2)

        # Sum of volumes: (B, C)
        # We use sum of probabilities + sum of targets (Soft Dice)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)

        # Dice Score: (B, C)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice Loss = 1 - Dice Score
        # Average over batch and classes
        dice_loss = 1.0 - dice_score.mean()

        # --- Combined Loss ---
        loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

        return loss
