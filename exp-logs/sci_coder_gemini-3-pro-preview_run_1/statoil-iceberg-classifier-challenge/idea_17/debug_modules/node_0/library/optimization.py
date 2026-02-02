import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from library.configuration import Config


class ConsistencyLoss(nn.Module):
    """
    Custom loss function implementing BCE with Label Smoothing and Geometric Consistency Regularization.

    Formula:
        L = BCEWithLogits(f(x), y_smooth) + lambda * ||sigma(f(x)) - sigma(f(x'))||^2

    Where:
        f(x) is the output for the first view.
        f(x') is the output for the second view.
        sigma is the sigmoid function.
        lambda is the consistency weight.
    """

    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        self.smoothing = Config.LABEL_SMOOTHING
        self.consistency_weight = Config.CONSISTENCY_LOSS_WEIGHT

    def forward(self, logits1, logits2, targets):
        """
        Args:
            logits1 (torch.Tensor): Logits from the first augmented view (Batch, 1).
            logits2 (torch.Tensor): Logits from the second augmented view (Batch, 1).
            targets (torch.Tensor): Ground truth labels (Batch,).
        """
        # Ensure targets are the correct shape (Batch, 1)
        targets = targets.view(-1, 1)

        # 1. Label Smoothing
        # y_smooth = y * (1 - epsilon) + 0.5 * epsilon
        with torch.no_grad():
            targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing

        # 2. BCE Loss on the first view
        # We use binary_cross_entropy_with_logits for numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(logits1, targets_smooth)

        # 3. Consistency Loss (MSE between probabilities)
        probs1 = torch.sigmoid(logits1)
        probs2 = torch.sigmoid(logits2)
        consistency_loss = F.mse_loss(probs1, probs2)

        # 4. Total Loss
        total_loss = bce_loss + self.consistency_weight * consistency_loss

        return total_loss


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
        verbose=True,
    )
    return scheduler
