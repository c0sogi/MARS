import torch
import torch.nn as nn
from library.config import Config


class WeightedBCELoss(nn.Module):
    """
    Binary Cross Entropy Loss with Inverse Class Frequency Weighting.
    Wraps nn.BCEWithLogitsLoss to handle logits and class imbalance.
    """

    def __init__(self, pos_weight_value=None, device=Config.DEVICE):
        """
        Args:
            pos_weight_value (float, optional): The weight for the positive class.
                                              Typically N_neg / N_pos.
            device (str): Device to place the weight tensor on.
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = None
        if pos_weight_value is not None:
            # pos_weight must be a tensor of size [1] for binary classification
            self.pos_weight = torch.tensor([pos_weight_value], device=device)

        # Initialize the underlying BCEWithLogitsLoss
        # reduction='mean' returns a scalar loss, which is required for the Mixup strategy described.
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model logits of shape (N, 1) or (N,)
            targets: Ground truth labels or soft targets of shape (N, 1) or (N,)
        """
        # Ensure targets are float (required for BCEWithLogitsLoss)
        targets = targets.float()

        # Ensure targets match input shape (e.g., handle (N,) vs (N, 1) mismatch)
        if inputs.shape != targets.shape:
            targets = targets.view_as(inputs)

        return self.criterion(inputs, targets)


class MixupLoss(nn.Module):
    """
    Computes the mixup loss given a criterion and mixed targets.
    Formula: lambda * loss(pred, y_a) + (1 - lambda) * loss(pred, y_b)
    """

    def __init__(self, criterion):
        """
        Args:
            criterion: The base loss function (e.g., WeightedBCELoss).
        """
        super(MixupLoss, self).__init__()
        self.criterion = criterion

    def forward(self, preds, target_a, target_b, lam):
        """
        Args:
            preds: Model predictions (logits)
            target_a: First set of targets
            target_b: Second set of targets
            lam: Lambda value from mixup (scalar)
        """
        # Calculate loss for both targets
        loss_a = self.criterion(preds, target_a)
        loss_b = self.criterion(preds, target_b)

        # Mix the scalar losses
        return lam * loss_a + (1 - lam) * loss_b
