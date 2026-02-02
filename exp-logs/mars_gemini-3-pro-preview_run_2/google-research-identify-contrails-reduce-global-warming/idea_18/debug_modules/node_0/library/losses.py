import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchDiceLoss(nn.Module):
    """
    Computes the Dice Loss over the entire batch (Global Dice).

    Instead of computing Dice per sample and averaging, this treats the
    entire batch as a single volume. This aligns with the 'Global Dice'
    metric used in evaluation and stabilizes gradients, especially when
    many samples in a batch might be empty (background only).
    """

    def __init__(self, smooth=1e-6):
        """
        Args:
            smooth (float): Smoothing factor to prevent division by zero.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid), shape (N, 1, H, W).
            targets (torch.Tensor): Ground truth masks, shape (N, 1, H, W).

        Returns:
            torch.Tensor: Scalar loss value (1 - Dice).
        """
        # Apply sigmoid to get probabilities [0, 1]
        probs = torch.sigmoid(logits)

        # Flatten the tensors: (N, 1, H, W) -> (N*H*W,)
        # We flatten the entire batch to compute global statistics
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Compute Intersection
        intersection = (probs_flat * targets_flat).sum()

        # Compute Union (Sum of probabilities + Sum of ground truth)
        union = probs_flat.sum() + targets_flat.sum()

        # Dice Coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return Dice Loss
        return 1.0 - dice


class HybridLoss(nn.Module):
    """
    Combines Binary Cross Entropy (BCE) and Batch-level Dice Loss.

    L_total = bce_weight * L_BCE + dice_weight * L_Dice

    BCE provides smooth gradients for pixel-level classification.
    Dice optimizes the Intersection over Union directly and handles class imbalance.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1e-6):
        """
        Args:
            bce_weight (float): Weight for the BCE component.
            dice_weight (float): Weight for the Dice component.
            smooth (float): Smoothing factor for Dice loss.
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        # BCEWithLogitsLoss combines Sigmoid and BCE for numerical stability
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = BatchDiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs, shape (N, 1, H, W).
            targets (torch.Tensor): Ground truth masks, shape (N, 1, H, W).

        Returns:
            torch.Tensor: Weighted combined loss.
        """
        loss = 0.0

        if self.bce_weight > 0:
            loss += self.bce_weight * self.bce_loss(logits, targets)

        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice_loss(logits, targets)

        return loss
