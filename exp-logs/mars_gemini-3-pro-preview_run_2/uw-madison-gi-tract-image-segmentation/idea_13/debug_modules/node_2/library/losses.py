import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BCETverskyLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Tversky Loss.
    Designed to handle Deep Supervision outputs from U-Net++.
    """

    def __init__(self):
        super(BCETverskyLoss, self).__init__()
        self.bce_weight = Config.BCE_WEIGHT
        self.tversky_weight = Config.TVERSKY_WEIGHT
        self.alpha = Config.TVERSKY_ALPHA
        self.beta = Config.TVERSKY_BETA
        self.smooth = Config.TVERSKY_SMOOTH

    def tversky_loss(self, y_pred_logits, y_true):
        """
        Calculates Tversky Loss.
        Args:
            y_pred_logits: Network output logits (B, C, H, W)
            y_true: Ground truth masks (B, C, H, W)
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(y_pred_logits)

        # Flatten label and prediction tensors
        probs = probs.view(-1)
        y_true = y_true.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (probs * y_true).sum()
        FP = ((1 - y_true) * probs).sum()
        FN = (y_true * (1 - probs)).sum()

        # Tversky Index
        tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        return 1.0 - tversky_index

    def forward(self, y_pred, y_true):
        """
        Forward pass.
        Args:
            y_pred: Model output. Can be a tensor or a list of tensors (Deep Supervision).
            y_true: Ground truth tensor.
        """
        # Handle Deep Supervision (list of outputs)
        if isinstance(y_pred, list) or isinstance(y_pred, tuple):
            loss = 0.0
            for prediction in y_pred:
                # Calculate BCE
                bce = F.binary_cross_entropy_with_logits(prediction, y_true)

                # Calculate Tversky
                tversky = self.tversky_loss(prediction, y_true)

                # Weighted sum
                loss += (self.bce_weight * bce) + (self.tversky_weight * tversky)

            # Average loss over all deep supervision outputs
            return loss / len(y_pred)

        else:
            # Standard single output
            bce = F.binary_cross_entropy_with_logits(y_pred, y_true)
            tversky = self.tversky_loss(y_pred, y_true)

            return (self.bce_weight * bce) + (self.tversky_weight * tversky)
