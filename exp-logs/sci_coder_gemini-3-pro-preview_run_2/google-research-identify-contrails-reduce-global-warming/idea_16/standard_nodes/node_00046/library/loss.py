import torch
import torch.nn as nn
from library.config import Config
from library.utils import dice_coef


class BatchDiceLoss(nn.Module):
    """
    Computes the Dice Loss over the entire batch (Global Dice).

    It applies a sigmoid activation to the logits to obtain probabilities,
    then uses the library.utils.dice_coef function which flattens the
    entire batch to compute the metric. The loss is defined as 1 - DiceScore.
    """

    def __init__(self, smooth=1e-6):
        super(BatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Calculate Dice coefficient
        # dice_coef flattens inputs internally, treating the batch as a single volume
        dice_score = dice_coef(probs, targets, self.smooth)

        # Return Dice Loss
        return 1.0 - dice_score


class HybridLoss(nn.Module):
    """
    Composite loss function combining Binary Cross Entropy and Batch Dice Loss.

    Formula:
        L_total = (w_bce * L_bce) + (w_dice * L_dice)

    Weights and smoothing parameters are sourced from the Config.
    """

    def __init__(self):
        super(HybridLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = BatchDiceLoss(smooth=Config.SMOOTH)

        self.bce_weight = Config.LOSS_BCE_WEIGHT
        self.dice_weight = Config.LOSS_DICE_WEIGHT

    def forward(self, logits, targets):
        # Compute individual losses
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice_loss(logits, targets)

        # Combine losses
        total_loss = (self.bce_weight * loss_bce) + (self.dice_weight * loss_dice)

        return total_loss
