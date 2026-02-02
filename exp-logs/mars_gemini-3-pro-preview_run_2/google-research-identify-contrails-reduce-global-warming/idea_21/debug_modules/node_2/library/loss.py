import torch
import torch.nn as nn


class BatchDiceLoss(nn.Module):
    """
    Computes the Dice Loss over the entire batch treated as a single volume.

    This stabilizes the gradient for sparse targets (like contrails) by aggregating
    statistics across all samples in the batch, rather than averaging sample-wise Dice scores.
    """

    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth (float): Smoothing factor to prevent division by zero.
        """
        super(BatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Raw model predictions (before sigmoid) of shape (B, C, H, W).
            targets (torch.Tensor): Ground truth binary masks of shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value (1 - Dice).
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors to treat the batch as a single volume
        # Shape becomes (N,) where N = B * C * H * W
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Compute intersection and cardinality
        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()

        # Compute Dice coefficient
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Return Dice Loss
        return 1.0 - dice_score


class HybridLoss(nn.Module):
    """
    Combines Binary Cross Entropy (BCE) and Batch-Level Dice Loss.

    Formula: L_total = L_BCE + L_BatchDice

    - BCE provides smooth, convex gradients for pixel-level classification.
    - BatchDice directly optimizes the evaluation metric and handles class imbalance.
    """

    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth (float): Smoothing factor passed to BatchDiceLoss.
        """
        super(HybridLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = BatchDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Raw model predictions of shape (B, C, H, W).
            targets (torch.Tensor): Ground truth binary masks of shape (B, C, H, W).

        Returns:
            torch.Tensor: Combined scalar loss.
        """
        # Calculate BCE Loss
        loss_bce = self.bce_loss(logits, targets)

        # Calculate Batch Dice Loss
        loss_dice = self.dice_loss(logits, targets)

        # Combine losses
        return loss_bce + loss_dice
