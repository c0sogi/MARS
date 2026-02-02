import torch
import torch.nn as nn
from library.config import Config


class BornAgainLoss(nn.Module):
    """
    Custom loss function for the Recursive 'Born-Again' Heterogeneous Ensemble strategy.

    This loss combines:
    1. Hard Label Loss: Standard BCEWithLogitsLoss against ground truth labels.
    2. Soft Label Loss: BCEWithLogitsLoss against soft targets (probabilities) from a previous generation.

    Both components use 'pos_weight' to handle class imbalance, as specified in the strategy:
    L = BCE(Target, Pred, pos_weight) + lambda * BCE(SoftTarget, Pred, pos_weight)
    """

    def __init__(self, pos_weight=None, distillation_lambda=None):
        """
        Args:
            pos_weight (torch.Tensor, optional): Tensor of weights for positive examples
                                                 (shape: [num_classes]). Used to handle class imbalance.
            distillation_lambda (float, optional): Weighting factor for the distillation loss term.
                                                   If None, defaults to Config.DISTILLATION_LAMBDA.
        """
        super(BornAgainLoss, self).__init__()

        self.distillation_lambda = (
            distillation_lambda
            if distillation_lambda is not None
            else Config.DISTILLATION_LAMBDA
        )

        # Initialize the criteria
        # We use separate instances conceptually, though they are identical in configuration.
        # BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class,
        # which is more numerically stable than using a plain Sigmoid followed by a BCELoss.
        if pos_weight is not None:
            self.hard_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.soft_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.hard_loss_fn = nn.BCEWithLogitsLoss()
            self.soft_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets, soft_targets=None):
        """
        Calculates the combined loss.

        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid), shape [batch_size, num_classes].
            targets (torch.Tensor): Ground truth binary labels, shape [batch_size, num_classes].
            soft_targets (torch.Tensor, optional): Soft targets (probabilities) from the teacher model,
                                                   shape [batch_size, num_classes].
                                                   If None, only hard loss is computed (Generation 0).

        Returns:
            torch.Tensor: The calculated loss scalar.
        """
        # Ensure targets are float for BCEWithLogitsLoss
        targets = targets.float()

        # 1. Calculate Hard Label Loss (Generation 0, 1, 2)
        hard_loss = self.hard_loss_fn(logits, targets)

        # 2. Calculate Soft Label Loss (Generation 1, 2 only)
        if soft_targets is not None:
            # Soft targets are probabilities (0-1), BCEWithLogitsLoss can handle continuous targets.
            # Note: logits are raw scores, soft_targets are probabilities.
            # BCEWithLogitsLoss(input, target) computes:
            # - target * log(sigmoid(input)) - (1 - target) * log(1 - sigmoid(input))
            soft_loss = self.soft_loss_fn(logits, soft_targets)

            # Combine losses
            total_loss = hard_loss + (self.distillation_lambda * soft_loss)
            return total_loss

        # If no soft targets provided (Generation 0), return only hard loss
        return hard_loss
