import torch
import torch.nn as nn
from library.config import Config


class BCEDiceLoss(nn.Module):
    """
    Implements a combined Binary Cross Entropy (BCE) and Dice Loss function.

    This loss function is designed to handle class imbalance and optimize for
    segmentation overlap. As per the experimental protocol, it uses a balanced
    Dice formulation (F1-score based) rather than a weighted Tversky approach,
    combined with BCE for stable convergence.
    """

    def __init__(self, smooth=Config.SMOOTH):
        """
        Args:
            smooth (float): Smoothing factor to prevent division by zero in Dice calculation.
        """
        super(BCEDiceLoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        Computes the combined BCE + Dice Loss.

        Args:
            inputs (torch.Tensor): Logits from the model of shape (Batch, Channels, Height, Width).
            targets (torch.Tensor): Binary ground truth masks of shape (Batch, Height, Width)
                                    or (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Ensure targets have the channel dimension if missing (e.g., (B, H, W) -> (B, 1, H, W))
        if inputs.dim() > targets.dim():
            targets = targets.unsqueeze(1)

        # Ensure targets are float for BCE calculation
        targets = targets.float()

        # ---------------------------------------------------------------------
        # 1. Binary Cross Entropy Loss
        # ---------------------------------------------------------------------
        # Provides pixel-wise supervision and helps with initial convergence
        bce_loss = self.bce(inputs, targets)

        # ---------------------------------------------------------------------
        # 2. Dice Loss (Balanced)
        # ---------------------------------------------------------------------
        # Optimizes for the overlap metric.
        # We use sigmoid to convert logits to probabilities [0, 1]
        probs = torch.sigmoid(inputs)

        # Flatten tensors to calculate intersection and union over the spatial dimensions
        # Shape: (Batch, N) where N = Channels * Height * Width
        batch_size = probs.shape[0]
        probs_flat = probs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)

        # Calculate Intersection and Union
        intersection = (probs_flat * targets_flat).sum(dim=1)

        # Denominator for balanced Dice is Sum(Probs) + Sum(Targets)
        # This corresponds to the F1-score formulation (Beta=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice Loss is 1 - Dice Score
        dice_loss = 1.0 - dice_score.mean()

        # ---------------------------------------------------------------------
        # Combined Loss
        # ---------------------------------------------------------------------
        return bce_loss + dice_loss
