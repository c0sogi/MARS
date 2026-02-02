import torch
import torch.nn as nn
from library.config import Config


class BatchDiceLoss(nn.Module):
    """
    Calculates the Dice Loss over the entire batch as a single volume.

    This aligns with the Global Dice metric used in evaluation.
    It applies a sigmoid activation to the logits to obtain probabilities,
    then flattens the batch to compute the intersection and cardinality.
    """

    def __init__(self, smooth: float = Config.SMOOTH):
        super(BatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid).
                                   Shape (B, C, H, W) or (B, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks.
                                    Shape (B, C, H, W) or (B, 1, H, W).

        Returns:
            torch.Tensor: Scalar loss value (1 - Dice).
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors to treat the batch as a single volume
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate intersection
        intersection = (probs_flat * targets_flat).sum()

        # Calculate cardinality (sum of probabilities + sum of true pixels)
        cardinality = probs_flat.sum() + targets_flat.sum()

        # Compute Soft Dice Score
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Return Dice Loss
        return 1.0 - dice_score


class HybridLoss(nn.Module):
    """
    Combines Binary Cross Entropy (BCE) and Batch-Level Dice Loss.

    BCE provides smooth, convex gradients for pixel-level classification,
    while BatchDiceLoss optimizes the global structural metric directly.
    """

    def __init__(
        self,
        bce_weight: float = Config.BCE_WEIGHT,
        dice_weight: float = Config.DICE_WEIGHT,
        smooth: float = Config.SMOOTH,
    ):
        super(HybridLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        # Initialize components
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = BatchDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the weighted sum of BCE and Dice loss.

        Args:
            logits (torch.Tensor): Raw model outputs.
            targets (torch.Tensor): Ground truth masks.

        Returns:
            torch.Tensor: The combined loss value.
        """
        # Ensure targets are float for BCE and Dice calculation
        targets = targets.float()

        # Compute individual losses
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        # Combine
        total_loss = (self.bce_weight * bce) + (self.dice_weight * dice)

        return total_loss
