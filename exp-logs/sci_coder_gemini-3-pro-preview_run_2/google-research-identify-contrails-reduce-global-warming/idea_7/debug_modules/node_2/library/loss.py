import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchDiceLoss(nn.Module):
    """
    Computes the Dice Loss over the entire flattened batch.

    This treats the batch as a single volume, which helps stabilize gradients
    when targets are sparse (small objects like contrails) and prevents
    div-by-zero or high variance issues associated with image-wise averaging.
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
            logits (torch.Tensor): Raw model outputs (before sigmoid). Shape (B, C, H, W).
            targets (torch.Tensor): Ground truth binary masks. Shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value (1 - Dice).
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors to treat the batch as a single volume
        # shape: (N,)
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
    Combines Binary Cross Entropy (BCE) and Batch Dice Loss.

    BCE provides smooth gradients for pixel-level classification, while
    Dice Loss directly optimizes the overlap metric used for evaluation.
    """

    def __init__(
        self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6
    ):
        """
        Args:
            bce_weight (float): Weight for the BCE component.
            dice_weight (float): Weight for the Dice component.
            smooth (float): Smoothing factor for the Dice loss.
        """
        super(HybridLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        # BCEWithLogitsLoss includes Sigmoid layer, more numerically stable
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = BatchDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Raw model outputs. Shape (B, C, H, W).
            targets (torch.Tensor): Ground truth binary masks. Shape (B, C, H, W).

        Returns:
            torch.Tensor: Weighted sum of BCE and Dice loss.
        """
        loss_bce = self.bce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)

        total_loss = (self.bce_weight * loss_bce) + (self.dice_weight * loss_dice)

        return total_loss
