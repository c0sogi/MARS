import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from library.configuration import Config


class IcebergLoss(nn.Module):
    """
    Standard BCE Loss with Label Smoothing.
    Replaces ConsistencyLoss to prioritize discriminative learning (Cite Lesson 00066).
    """

    def __init__(self):
        super(IcebergLoss, self).__init__()
        self.smoothing = Config.LABEL_SMOOTHING
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Logits (Batch, 1).
            targets (torch.Tensor): Ground truth labels (Batch,).
        """
        targets = targets.view(-1, 1)

        # Label Smoothing
        with torch.no_grad():
            targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing

        return self.bce(logits, targets_smooth)


def get_optimizer(model):
    """
    Creates the AdamW optimizer.
    """
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    return optimizer


def get_scheduler(optimizer):
    """
    Creates the ReduceLROnPlateau scheduler.
    """
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )
    return scheduler
