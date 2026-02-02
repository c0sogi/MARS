import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """
    Tversky Loss for segmentation tasks.

    The Tversky index is a generalization of the Dice coefficient and Jaccard index.
    It adds weights (alpha and beta) to False Positives and False Negatives.

    Formula:
        T = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        Loss = 1 - T

    Where:
        TP = True Positives
        FP = False Positives
        FN = False Negatives
        alpha = weight for False Positives
        beta = weight for False Negatives

    If alpha=beta=0.5, it is equivalent to Dice Loss.
    If alpha=beta=1.0, it is equivalent to Jaccard (IoU) Loss.
    """

    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model predictions (logits) of shape (B, C, H, W).
            targets (torch.Tensor): Ground truth masks of shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Apply sigmoid to convert logits to probabilities
        inputs = torch.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = inputs.flatten()
        targets = targets.flatten()

        # Calculate True Positives, False Positives, False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        # Calculate Tversky Index
        tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        return 1.0 - tversky_index


class BCETverskyLoss(nn.Module):
    """
    Combination of Binary Cross Entropy (BCE) and Tversky Loss.

    BCE provides smooth gradients for pixel-wise classification, while Tversky
    optimizes the overlap metric directly and handles class imbalance via alpha/beta.
    """

    def __init__(
        self, alpha=0.5, beta=0.5, smooth=1.0, bce_weight=0.5, tversky_weight=0.5
    ):
        super(BCETverskyLoss, self).__init__()
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight

        # BCEWithLogitsLoss includes Sigmoid layer for numerical stability
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model predictions (logits) of shape (B, C, H, W).
            targets (torch.Tensor): Ground truth masks of shape (B, C, H, W).

        Returns:
            torch.Tensor: Weighted sum of BCE and Tversky loss.
        """
        bce_loss = self.bce(inputs, targets)
        tversky_loss = self.tversky(inputs, targets)

        return (self.bce_weight * bce_loss) + (self.tversky_weight * tversky_loss)
