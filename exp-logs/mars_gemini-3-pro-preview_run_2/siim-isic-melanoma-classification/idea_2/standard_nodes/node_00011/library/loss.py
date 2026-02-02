import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class WeightedBCE(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for imbalanced classification.
    Wraps nn.BCEWithLogitsLoss to handle reshaping and device placement of pos_weight.
    """

    def __init__(self, pos_weight=Config.POS_WEIGHT):
        super(WeightedBCE, self).__init__()
        # We create the criterion. The pos_weight tensor needs to be passed.
        # BCEWithLogitsLoss will register it as a buffer.
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, inputs, targets):
        # Ensure inputs and targets are of the same shape (N, 1)
        inputs = inputs.view(-1, 1)
        targets = targets.view(-1, 1).float()

        return self.criterion(inputs, targets)
