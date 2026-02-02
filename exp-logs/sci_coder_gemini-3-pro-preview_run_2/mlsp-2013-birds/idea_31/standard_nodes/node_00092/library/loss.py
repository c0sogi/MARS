import torch
import torch.nn as nn
from library.config import Config


class DistillationLoss(nn.Module):
    """
    Custom loss function for TTA-Enhanced Recursive Distillation.

    Computes a weighted sum of:
    1. Standard Binary Cross Entropy with Logits (against hard ground truth).
    2. Distillation Binary Cross Entropy with Logits (against soft TTA-enhanced targets).

    Both terms utilize 'pos_weight' to handle class imbalance, ensuring that
    rare classes are not ignored even in the soft-target distillation phase.
    """

    def __init__(self, pos_weight=None, distillation_lambda=None):
        """
        Args:
            pos_weight (torch.Tensor, optional): A weight of positive examples.
                                                 Must be a vector with length equal to the number of classes.
            distillation_lambda (float, optional): Weighting factor for the distillation loss term.
                                                   Defaults to Config.DISTILLATION_LAMBDA.
        """
        super(DistillationLoss, self).__init__()

        self.pos_weight = pos_weight
        self.distillation_lambda = (
            distillation_lambda
            if distillation_lambda is not None
            else Config.DISTILLATION_LAMBDA
        )

        # Initialize the base criterion.
        # BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class.
        # It is numerically more stable than using a plain Sigmoid followed by a BCELoss.
        # We pass pos_weight here; it will be registered as a buffer and move to device automatically.
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, student_logits, targets, soft_targets=None):
        """
        Args:
            student_logits (torch.Tensor): Raw output logits from the student model. Shape: (Batch, Num_Classes)
            targets (torch.Tensor): Binary ground truth labels. Shape: (Batch, Num_Classes)
            soft_targets (torch.Tensor, optional): Probability targets from the teacher/ensemble.
                                                   Shape: (Batch, Num_Classes).
                                                   If None, only hard label loss is computed.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Compute standard supervised loss against hard labels
        loss_hard = self.criterion(student_logits, targets)

        # 2. If soft targets are provided, compute distillation loss
        if soft_targets is not None:
            # Note: BCEWithLogitsLoss takes logits as input and probabilities as target.
            # student_logits are logits.
            # soft_targets are probabilities (0-1) derived from TTA averaging.
            # We apply the same pos_weight to ensure rare class knowledge is preserved.
            loss_soft = self.criterion(student_logits, soft_targets)

            # Combine losses
            total_loss = loss_hard + (self.distillation_lambda * loss_soft)
            return total_loss

        return loss_hard
