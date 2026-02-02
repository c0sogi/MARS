import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BCETverskyLoss(nn.Module):
    """
    A combined loss function that uses Binary Cross Entropy (BCE) and Tversky Loss.
    Useful for imbalanced segmentation tasks where False Negatives need to be penalized
    differently than False Positives.
    """

    def __init__(
        self,
        bce_weight=Config.LOSS_BCE_WEIGHT,
        tversky_weight=Config.LOSS_TVERSKY_WEIGHT,
        alpha=Config.TVERSKY_ALPHA,
        beta=Config.TVERSKY_BETA,
        smooth=Config.TVERSKY_SMOOTH,
    ):
        """
        Args:
            bce_weight (float): Weight for the BCE component.
            tversky_weight (float): Weight for the Tversky component.
            alpha (float): Weight for False Positives in Tversky index.
            beta (float): Weight for False Negatives in Tversky index.
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super(BCETverskyLoss, self).__init__()
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

        # BCEWithLogitsLoss is more numerically stable than Sigmoid + BCELoss
        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, inputs, targets):
        """
        Computes the combined loss.

        Args:
            inputs (torch.Tensor): Model predictions (logits) of shape (Batch, Channels, Height, Width).
            targets (torch.Tensor): Ground truth masks of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # --- BCE Loss ---
        # inputs are logits, targets should be float
        bce_loss = self.bce_loss_fn(inputs, targets.float())

        # --- Tversky Loss ---
        # Apply sigmoid to get probabilities for Tversky calculation
        probs = torch.sigmoid(inputs)

        # Flatten label and prediction tensors
        # We flatten to (Batch * Channels, -1) to compute metric per sample/channel or global
        # Here we flatten all dimensions except the first (Batch) to treat every image in batch
        # effectively, but usually for segmentation metrics we flatten everything to get a global count
        # or flatten per sample.
        # To align with standard implementations: flatten to 1D vectors.
        probs = probs.reshape(-1)
        targets = targets.reshape(-1)

        # True Positives, False Positives, False Negatives
        TP = (probs * targets).sum()
        FP = ((1 - targets) * probs).sum()
        FN = (targets * (1 - probs)).sum()

        # Tversky Index
        tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        tversky_loss = 1.0 - tversky_index

        # --- Combined Loss ---
        total_loss = (self.bce_weight * bce_loss) + (self.tversky_weight * tversky_loss)

        return total_loss
