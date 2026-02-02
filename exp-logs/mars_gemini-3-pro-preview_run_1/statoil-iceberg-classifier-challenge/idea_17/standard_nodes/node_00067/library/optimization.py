import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from library.configuration import Config


class SoftBCELoss(nn.Module):
    """
    BCEWithLogitsLoss with Label Smoothing.
    Reverts to standard supervised loss as Consistency Regularization degraded performance (Cite 00066).
    """

    def __init__(self):
        super(SoftBCELoss, self).__init__()
        self.smoothing = Config.LABEL_SMOOTHING

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Logits (Batch, 1).
            targets (torch.Tensor): Ground truth labels (Batch,).
        """
        # Ensure targets are the correct shape (Batch, 1)
        targets = targets.view(-1, 1)

        # Label Smoothing
        # y_smooth = y * (1 - epsilon) + 0.5 * epsilon
        with torch.no_grad():
            targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing

        # BCE Loss
        loss = F.binary_cross_entropy_with_logits(logits, targets_smooth)

        return loss


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
