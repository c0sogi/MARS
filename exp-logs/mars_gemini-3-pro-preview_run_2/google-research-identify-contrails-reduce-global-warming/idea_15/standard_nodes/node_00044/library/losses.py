import torch
import torch.nn as nn


class BatchDiceLoss(nn.Module):
    """
    Calculates the Dice Loss across the entire flattened batch.

    This treats the whole batch as a single volume for the purpose of calculating
    the Dice score, which helps stabilize gradients when foreground objects (contrails)
    are sparse or absent in many images within a batch.

    Formula:
        Dice = (2 * |X n Y| + smooth) / (|X| + |Y| + smooth)
        Loss = 1 - Dice
    """

    def __init__(self, smooth=1e-6):
        """
        Args:
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super(BatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (torch.Tensor): Raw logits from the model of shape (B, C, H, W).
            y_true (torch.Tensor): Ground truth binary masks of shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Apply sigmoid to logits to get probabilities in [0, 1]
        probs = torch.sigmoid(y_pred)

        # Flatten the tensors: (B, C, H, W) -> (N,)
        # This aggregates statistics over the entire batch
        probs_flat = probs.view(-1)
        targets_flat = y_true.view(-1)

        # Calculate intersection and union
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return Dice Loss
        return 1.0 - dice


class HybridLoss(nn.Module):
    """
    Combines Binary Cross Entropy (BCE) Loss and Batch Dice Loss.

    BCE provides smooth gradients for pixel-level classification, while
    Batch Dice Loss optimizes directly for the overlap metric and handles
    class imbalance effectively.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        """
        Args:
            bce_weight (float): Weight for the BCE component.
            dice_weight (float): Weight for the Dice component.
            smooth (float): Smoothing factor for the Dice loss.
        """
        super(HybridLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

        # Initialize component losses
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.dice_loss_fn = BatchDiceLoss(smooth=self.smooth)

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (torch.Tensor): Raw logits from the model.
            y_true (torch.Tensor): Ground truth binary masks.

        Returns:
            torch.Tensor: Weighted sum of BCE and Dice losses.
        """
        # Ensure targets are float for BCE calculation
        y_true = y_true.float()

        loss_bce = self.bce_loss_fn(y_pred, y_true)
        loss_dice = self.dice_loss_fn(y_pred, y_true)

        total_loss = (self.bce_weight * loss_bce) + (self.dice_weight * loss_dice)

        return total_loss
