import torch
import torch.nn as nn
from library.config import Config


class TverskyLoss(nn.Module):
    """
    Tversky Loss for imbalanced segmentation tasks.

    The Tversky index is a generalization of the Dice coefficient that allows for
    flexibility in balancing False Positives (FP) and False Negatives (FN).
    This is particularly useful for the F0.5 metric, which weights precision
    higher than recall.

    Formula:
        Score = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        Loss = 1 - Score

    Attributes:
        alpha (float): Weight penalizing False Positives.
        beta (float): Weight penalizing False Negatives.
        smooth (float): Smoothing factor for numerical stability.
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
            inputs (torch.Tensor): Predicted probabilities from the model.
                                   Shape: (Batch, Channels, Height, Width) or (Batch, Height, Width).
                                   Values should be in range [0, 1].
            targets (torch.Tensor): Ground truth binary masks.
                                    Shape: Same as inputs.
                                    Values should be 0 or 1.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Flatten label and prediction tensors to compute global stats across the batch
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
        # inputs are probabilities, targets are binary
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        # Calculate Tversky Index
        tversky_score = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        return 1 - tversky_score
