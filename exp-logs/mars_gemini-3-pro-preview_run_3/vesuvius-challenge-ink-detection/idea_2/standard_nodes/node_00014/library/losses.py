import torch
import torch.nn as nn
from library.config import Config


class BCEDiceLoss(nn.Module):
    """
    Balanced combination of Binary Cross Entropy and Dice Loss.
    Cite {solution_lesson_node_00011}: Balanced loss often outperforms skewed losses for F0.5.
    """

    def __init__(self, bce_weight=Config.BCE_WEIGHT):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCELoss()

    def forward(self, inputs, targets):
        # BCE
        bce_loss = self.bce(inputs, targets)

        # Dice
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + 1e-6) / (
            inputs_flat.sum() + targets_flat.sum() + 1e-6
        )
        dice_loss = 1 - dice_score

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
