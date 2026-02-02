import torch
import torch.nn as nn
from library.config import Config


class SequenceLoss(nn.Module):
    """
    Standard Cross Entropy Loss for Sequence Classification.
    Features: Class Weighting, Label Smoothing, Ignore Index.
    """

    def __init__(self, ignore_index=-100):
        super(SequenceLoss, self).__init__()

        # 1. Define Class Weights
        weights = torch.ones(Config.NUM_CLASSES)
        if Config.NUM_CLASSES > 0:
            weights[0] = Config.BACKGROUND_WEIGHT

        # 2. Initialize CrossEntropyLoss
        self.criterion = nn.CrossEntropyLoss(
            weight=weights,
            ignore_index=ignore_index,
            label_smoothing=Config.LABEL_SMOOTHING,
            reduction="mean",
        )

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): (Batch, Time, NumClasses)
            targets (torch.Tensor): (Batch, Time)
        """
        # Flatten: (B*T, C) and (B*T)
        logits_flat = logits.view(-1, Config.NUM_CLASSES)
        targets_flat = targets.view(-1)

        return self.criterion(logits_flat, targets_flat)
