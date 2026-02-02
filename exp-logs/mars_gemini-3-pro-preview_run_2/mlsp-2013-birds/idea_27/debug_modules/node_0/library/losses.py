import torch
import torch.nn as nn
from library.config import Config


class WeightedDistillationLoss(nn.Module):
    """
    Implements the Weighted Self-Distillation Loss as defined in the Born-Again strategy.

    The loss function consists of two components:
    1. Hard Label Loss: BCEWithLogitsLoss against ground truth binary labels.
    2. Distillation Loss: BCEWithLogitsLoss against soft targets (probabilities) from the teacher ensemble.

    Both components utilize `pos_weight` to handle class imbalance.

    Formula:
        L = BCE(Target, Pred, pos_weight) + lambda * BCE(Soft_Target, Pred, pos_weight)
    """

    def __init__(self, pos_weight=None, distillation_lambda=None):
        """
        Args:
            pos_weight (torch.Tensor, optional): Tensor of weights for positive classes (num_classes,).
                                                 Used to handle class imbalance in both hard and soft loss terms.
                                                 Should be on the same device as the model/logits.
            distillation_lambda (float, optional): Weighting factor for the distillation term.
                                                   If None, uses the value from Config.DISTILLATION_LAMBDA.
        """
        super(WeightedDistillationLoss, self).__init__()

        if distillation_lambda is None:
            self.distillation_lambda = Config.DISTILLATION_LAMBDA
        else:
            self.distillation_lambda = distillation_lambda

        # Initialize BCEWithLogitsLoss.
        # This combines a Sigmoid layer and the BCELoss in one single class for numerical stability.
        # pos_weight allows trading off recall and precision by up-weighting the positive class.
        # It is registered as a buffer and will move to the device with the module.
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets, soft_targets=None):
        """
        Calculates the combined loss.

        Args:
            logits (torch.Tensor): Predicted logits from the student model. Shape: (Batch, Num_Classes).
            targets (torch.Tensor): Ground truth binary labels. Shape: (Batch, Num_Classes).
            soft_targets (torch.Tensor, optional): Soft targets (probabilities) from the teacher/ensemble.
                                                   Shape: (Batch, Num_Classes).
                                                   If None, only the hard label loss is computed.

        Returns:
            torch.Tensor: The scalar loss value.
        """
        # 1. Calculate Hard Label Loss (Standard BCE)
        # logits are raw scores, targets are 0/1
        loss_hard = self.bce_loss(logits, targets)

        # 2. Calculate Distillation Loss (if soft targets provided)
        if soft_targets is not None:
            # We explicitly apply the same class imbalance weights (pos_weight) to the distillation term.
            # This ensures the student learns "dark knowledge" about rare classes without being
            # overwhelmed by the majority negative class.
            loss_soft = self.bce_loss(logits, soft_targets)

            # Combine losses
            total_loss = loss_hard + (self.distillation_lambda * loss_soft)
            return total_loss

        return loss_hard
