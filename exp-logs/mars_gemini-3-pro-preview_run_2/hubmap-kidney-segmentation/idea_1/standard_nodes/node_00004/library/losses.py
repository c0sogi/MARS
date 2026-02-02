import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Optimizes the Dice coefficient directly, which is robust to class imbalance.
    """

    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the Soft Dice Loss.

        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid), shape (N, C, H, W).
            targets (torch.Tensor): Ground truth masks, shape (N, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value (mean over the batch).
        """
        # Apply sigmoid activation to logits to get probabilities [0, 1]
        probs = torch.sigmoid(logits)

        # Ensure targets are float for calculation
        targets = targets.float()

        # Flatten spatial dimensions (H, W) to (N, -1)
        # This calculates the Dice score per image in the batch
        batch_size = logits.size(0)
        probs_flat = probs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)

        # Calculate Intersection and Union
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        # Calculate Dice Coefficient
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Calculate Dice Loss (1 - Dice)
        loss = 1.0 - dice_score

        # Return the mean loss over the batch
        return loss.mean()
