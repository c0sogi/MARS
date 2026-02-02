import torch
import torch.nn as nn
from library.config import Config


class WeightedBCELoss(nn.Module):
    """
    Binary Cross Entropy Loss with inverse class frequency weighting.
    Wraps nn.BCEWithLogitsLoss to handle imbalanced datasets.
    """

    def __init__(self, pos_weight=None, device=Config.DEVICE):
        """
        Args:
            pos_weight (torch.Tensor, optional): Weight for the positive class.
                                                 Should be broadcastable to the output shape.
            device (str): Device to move the weight to.
        """
        super(WeightedBCELoss, self).__init__()

        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight)
            # Ensure weight is on the correct device
            pos_weight = pos_weight.to(device)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted logits. Shape (B, 1) or (B, C).
            targets (torch.Tensor): Ground truth labels. Shape (B,) or (B, 1).
        """
        # Ensure targets have the same shape as inputs (e.g., reshape (B,) to (B, 1))
        if targets.shape != inputs.shape:
            targets = targets.view_as(inputs)

        return self.criterion(inputs, targets)


class MixupLoss(nn.Module):
    """
    Wrapper for handling Mixup augmentation loss.
    Computes the loss by mixing the weighted scalar losses of the input pair
    rather than mixing the labels. This strategy preserves gradients from
    minority class samples more effectively.
    """

    def __init__(self, criterion):
        """
        Args:
            criterion (nn.Module): The base loss function (e.g., WeightedBCELoss).
        """
        super(MixupLoss, self).__init__()
        self.criterion = criterion

    def forward(self, preds, target_a, target_b, lam):
        """
        Args:
            preds (torch.Tensor): Model predictions (logits).
            target_a (torch.Tensor): Targets for the first image in the mixup pair.
            target_b (torch.Tensor): Targets for the second image in the mixup pair.
            lam (float): Mixup coefficient (lambda) used for mixing inputs.

        Returns:
            torch.Tensor: The combined loss.
        """
        # Compute loss for both targets individually
        loss_a = self.criterion(preds, target_a)
        loss_b = self.criterion(preds, target_b)

        # Combine scalar losses based on lambda
        return lam * loss_a + (1 - lam) * loss_b
