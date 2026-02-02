import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    TVERSKY_ALPHA,
    TVERSKY_BETA,
    TVERSKY_SMOOTH,
    BCE_WEIGHT,
    TVERSKY_WEIGHT,
)


class TverskyLoss(nn.Module):
    """
    Tversky Loss for segmentation tasks.

    The Tversky index is a generalization of the Dice coefficient and Jaccard index.
    It adds weights to False Positives (alpha) and False Negatives (beta).

    Formula:
        T = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        Loss = 1 - T
    """

    def __init__(self, alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA, smooth=TVERSKY_SMOOTH):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # y_pred: (Batch, Class, Height, Width) - Logits
        # y_true: (Batch, Class, Height, Width) - Binary 0/1

        # Apply sigmoid to get probabilities
        y_pred = torch.sigmoid(y_pred)

        # Flatten label and prediction tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (y_pred * y_true).sum()
        FP = ((1 - y_true) * y_pred).sum()
        FN = (y_true * (1 - y_pred)).sum()

        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        return 1 - tversky


class BCETverskyLoss(nn.Module):
    """
    Combined BCE and Tversky Loss.
    Supports Deep Supervision by handling list inputs.
    """

    def __init__(self, bce_weight=BCE_WEIGHT, tversky_weight=TVERSKY_WEIGHT):
        super(BCETverskyLoss, self).__init__()
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss()

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Model output. Can be a tensor (B, C, H, W) or a list of tensors
                    if Deep Supervision is enabled.
            y_true: Ground truth mask (B, C, H, W).
        """
        # Handle Deep Supervision (list of outputs)
        if isinstance(y_pred, (list, tuple)):
            loss = 0.0
            for prediction in y_pred:
                loss += self._compute_loss(prediction, y_true)
            # Average the loss over the number of outputs
            loss /= len(y_pred)
            return loss
        else:
            return self._compute_loss(y_pred, y_true)

    def _compute_loss(self, prediction, target):
        bce_loss = self.bce(prediction, target)
        tversky_loss = self.tversky(prediction, target)
        return (self.bce_weight * bce_loss) + (self.tversky_weight * tversky_loss)
