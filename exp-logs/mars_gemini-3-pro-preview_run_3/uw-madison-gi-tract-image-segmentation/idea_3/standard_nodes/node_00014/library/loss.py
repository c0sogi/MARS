import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BCETverskyLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Tversky Loss.

    This loss function is designed to address class imbalance in segmentation tasks.
    It combines the pixel-wise accuracy of BCE with the overlap-based Tversky index,
    which generalizes the Dice coefficient by allowing flexibility in penalizing
    False Positives (FP) and False Negatives (FN).
    """

    def __init__(self):
        super(BCETverskyLoss, self).__init__()
        # Load hyperparameters from Config
        self.alpha = Config.TVERSKY_ALPHA
        self.beta = Config.TVERSKY_BETA
        self.smooth = Config.TVERSKY_SMOOTH
        self.bce_weight = Config.BCE_WEIGHT
        self.tversky_weight = Config.TVERSKY_WEIGHT

    def forward(self, inputs, targets):
        """
        Forward pass for the loss calculation.

        Args:
            inputs (torch.Tensor): Model predictions (logits) of shape (B, C, H, W).
            targets (torch.Tensor): Ground truth binary masks of shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Ensure targets are float for numerical operations
        targets = targets.float()

        # 1. Binary Cross Entropy Loss
        # F.binary_cross_entropy_with_logits includes sigmoid activation internally for numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)

        # 2. Tversky Loss
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(inputs)

        # Flatten the tensors to calculate global metrics
        # Flattening treats all pixels in the batch as a single set, which is robust for batch-based optimization
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
        TP = (probs_flat * targets_flat).sum()
        FP = ((1.0 - targets_flat) * probs_flat).sum()
        FN = (targets_flat * (1.0 - probs_flat)).sum()

        # Calculate Tversky Index
        tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FN + self.beta * FP + self.smooth
        )

        # Tversky Loss is 1 - Tversky Index
        tversky_loss = 1.0 - tversky_index

        # Combine losses
        total_loss = (self.bce_weight * bce_loss) + (self.tversky_weight * tversky_loss)

        return total_loss
