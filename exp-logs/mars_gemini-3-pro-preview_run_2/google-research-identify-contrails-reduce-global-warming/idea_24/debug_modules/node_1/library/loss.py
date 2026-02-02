import torch
import torch.nn as nn


class BatchDiceLoss(nn.Module):
    """
    Computes the Dice Loss over the entire batch.

    Instead of computing Dice per sample and averaging, this flattens the
    entire batch (B, C, H, W) into a single vector. This stabilizes gradients,
    especially when individual samples might be empty or have very few positive pixels.
    """

    def __init__(self, smooth=1e-6):
        """
        Args:
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super(BatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs (before Sigmoid), shape (B, C, H, W).
            targets (torch.Tensor): Ground truth binary masks, shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value (1 - Dice).
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors to treat the batch as a single volume
        # Shape becomes (N,) where N = B * C * H * W
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate Intersection: |X n Y|
        intersection = (probs_flat * targets_flat).sum()

        # Calculate Cardinality: |X| + |Y|
        union = probs_flat.sum() + targets_flat.sum()

        # Compute Dice Coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return Dice Loss
        return 1.0 - dice


class HybridLoss(nn.Module):
    """
    Combines Binary Cross Entropy (BCE) and Batch-Level Dice Loss.

    L_total = weight_bce * L_BCE + weight_dice * L_BatchDice
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1e-6):
        """
        Args:
            bce_weight (float): Weight for the BCE component.
            dice_weight (float): Weight for the Dice component.
            smooth (float): Smoothing factor for the Dice loss.
        """
        super(HybridLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        # BCEWithLogitsLoss combines Sigmoid and BCE for numerical stability
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = BatchDiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs, shape (B, C, H, W).
            targets (torch.Tensor): Ground truth binary masks, shape (B, C, H, W).

        Returns:
            torch.Tensor: Weighted sum of BCE and Dice losses.
        """
        # Compute BCE Loss
        # Ensure targets are float for BCEWithLogitsLoss
        loss_bce = self.bce_loss(logits, targets)

        # Compute Batch Dice Loss
        loss_dice = self.dice_loss(logits, targets)

        # Combine losses
        total_loss = (self.bce_weight * loss_bce) + (self.dice_weight * loss_dice)

        return total_loss
