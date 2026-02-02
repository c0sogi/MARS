import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchDiceLoss(nn.Module):
    """
    Computes the Dice Loss over the entire flattened batch.

    This treats the whole batch as a single volume, which stabilizes gradients
    and aligns with the global Dice metric used in the competition.

    Formula: Loss = 1 - (2 * |X n Y| + smooth) / (|X| + |Y| + smooth)
    """

    def __init__(self, smooth=1.0):
        super(BatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid), shape (B, C, H, W).
            targets (torch.Tensor): Ground truth masks, shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors to treat the batch as a single volume
        # shape becomes (N,) where N = B * C * H * W
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate intersection and union
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Compute Dice coefficient
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return Dice Loss
        return 1.0 - dice_score


class HybridLoss(nn.Module):
    """
    Hybrid Loss combining Binary Cross Entropy and Batch-Level Dice Loss.

    L_total = (bce_weight * L_BCE) + (dice_weight * L_Dice)
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
        super(HybridLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        # Initialize component losses
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.dice_loss_fn = BatchDiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs, shape (B, C, H, W).
            targets (torch.Tensor): Ground truth masks, shape (B, C, H, W).

        Returns:
            torch.Tensor: Weighted combined loss.
        """
        # Ensure targets are float for BCE calculation
        if targets.dtype != logits.dtype:
            targets = targets.type_as(logits)

        # Calculate individual losses
        bce = self.bce_loss_fn(logits, targets)
        dice = self.dice_loss_fn(logits, targets)

        # Combine
        total_loss = (self.bce_weight * bce) + (self.dice_weight * dice)

        return total_loss
