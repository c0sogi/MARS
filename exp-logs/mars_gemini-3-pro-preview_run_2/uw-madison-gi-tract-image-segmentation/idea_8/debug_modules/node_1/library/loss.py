import torch
import torch.nn as nn
from library.config import Config


class TverskyLoss(nn.Module):
    """
    Implements Tversky Loss for semantic segmentation.

    The Tversky loss is a generalization of the Dice loss. It adds weights (alpha and beta)
    to False Positives and False Negatives, allowing the model to be tuned for precision
    or recall. This is particularly useful for imbalanced datasets where the region of
    interest (ROI) is small.

    Formula:
        T = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        Loss = 1 - T

    Args:
        alpha (float): Weight for False Positives.
        beta (float): Weight for False Negatives.
        smooth (float): Smoothing factor to prevent division by zero.
    """

    def __init__(
        self,
        alpha=Config.TVERSKY_ALPHA,
        beta=Config.TVERSKY_BETA,
        smooth=Config.TVERSKY_SMOOTH,
    ):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Calculates the Tversky loss.

        Args:
            inputs (torch.Tensor): Predicted logits with shape (Batch, Channels, Height, Width).
            targets (torch.Tensor): Ground truth binary masks with shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Apply sigmoid to convert logits to probabilities
        inputs = torch.sigmoid(inputs)

        # Flatten label and prediction tensors
        # We flatten spatial dimensions (H, W) but keep Batch and Channel dimensions distinct initially
        # or flatten everything depending on desired reduction.
        # Here we calculate loss per sample/channel and average.
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        # Tversky Index
        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        return 1 - tversky
