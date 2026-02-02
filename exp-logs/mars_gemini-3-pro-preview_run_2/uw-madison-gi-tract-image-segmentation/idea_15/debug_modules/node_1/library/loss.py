import torch
import torch.nn as nn
from library.config import Config


class TverskyLoss(nn.Module):
    """
    Calculates Tversky Loss, a generalization of Dice Loss that allows
    weighting False Positives and False Negatives differently.
    """

    def __init__(self, alpha, beta, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: logits from the model (B, C, H, W)
        # targets: binary ground truth masks (B, C, H, W)

        # Apply sigmoid to convert logits to probabilities
        inputs = torch.sigmoid(inputs)

        # Flatten label and prediction tensors to compute global stats
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        # Calculate True Positives, False Positives, False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        # Calculate Tversky Index
        tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        return 1 - tversky_index


class BCETverskyLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Tversky Loss.
    Handles Deep Supervision by aggregating losses from multiple decoder outputs.
    """

    def __init__(self):
        super(BCETverskyLoss, self).__init__()
        # Load hyperparameters from Config
        self.alpha = Config.TVERSKY_ALPHA
        self.beta = Config.TVERSKY_BETA
        self.smooth = Config.TVERSKY_SMOOTH
        self.bce_weight = Config.BCE_WEIGHT
        self.tversky_weight = Config.TVERSKY_WEIGHT

        # Initialize component losses
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(alpha=self.alpha, beta=self.beta, smooth=self.smooth)

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Prediction from model. Can be a tensor or a list of tensors (Deep Supervision).
            y_true: Ground truth mask.
        """
        # Handle Deep Supervision (list of tensors returned by U-Net++)
        if isinstance(y_pred, (list, tuple)):
            loss = 0.0
            for pred in y_pred:
                loss += self._compute_single_loss(pred, y_true)
            # Return average loss across all supervision heads
            return loss / len(y_pred)
        else:
            # Standard single output
            return self._compute_single_loss(y_pred, y_true)

    def _compute_single_loss(self, pred, target):
        """
        Computes the weighted sum of BCE and Tversky loss for a single prediction.
        """
        bce_loss = self.bce(pred, target)
        tversky_loss = self.tversky(pred, target)

        return (self.bce_weight * bce_loss) + (self.tversky_weight * tversky_loss)
