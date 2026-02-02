import torch
import torch.nn as nn
from library.config import Config


class WeightedBCE(nn.Module):
    """
    Weighted Binary Cross Entropy Loss.
    Wraps nn.BCEWithLogitsLoss with positive class weighting to handle imbalance.
    """

    def __init__(self, pos_weights: torch.Tensor):
        """
        Args:
            pos_weights (torch.Tensor): Tensor of shape (num_classes,) containing weights for positive examples.
                                        Should be on the appropriate device.
        """
        super(WeightedBCE, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the weighted BCE loss.

        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid), shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels (0 or 1), shape (batch_size, num_classes).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        return self.criterion(logits, targets)


class DistillationLoss(nn.Module):
    """
    Weighted Distillation Loss for the Born-Again Ensemble strategy.
    Combines supervised loss (Weighted BCE) and distillation loss (Weighted BCE against soft targets).

    Formula: L = BCE(Target, Pred, w) + lambda * BCE(Soft_Target, Pred, w)

    Applying pos_weights to the distillation term ensures that the student model learns
    from the teacher's soft probabilities regarding rare classes effectively.
    """

    def __init__(
        self,
        pos_weights: torch.Tensor,
        lambda_distill: float = Config.DISTILLATION_LAMBDA,
    ):
        """
        Args:
            pos_weights (torch.Tensor): Tensor of shape (num_classes,) containing weights for positive examples.
            lambda_distill (float): Weighting factor for the distillation term. Defaults to Config.DISTILLATION_LAMBDA.
        """
        super(DistillationLoss, self).__init__()
        self.lambda_distill = lambda_distill

        # Use the same weighted criterion for both supervised and distillation parts
        # as per the strategy to handle class imbalance in both signals.
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    def forward(
        self,
        logits: torch.Tensor,
        hard_targets: torch.Tensor,
        soft_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the combined supervised and distillation loss.

        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid), shape (batch_size, num_classes).
            hard_targets (torch.Tensor): Ground truth labels (0 or 1), shape (batch_size, num_classes).
            soft_targets (torch.Tensor): Soft probabilities from teacher/TTA, shape (batch_size, num_classes).
                                         These should be probabilities (0-1), not logits.

        Returns:
            torch.Tensor: Combined scalar loss value.
        """
        # Supervised loss: Model logits vs Ground Truth
        supervised_loss = self.criterion(logits, hard_targets)

        # Distillation loss: Model logits vs Soft Targets
        # BCEWithLogitsLoss accepts soft targets (probabilities) as the target argument.
        # It applies sigmoid to the input logits internally.
        distillation_loss = self.criterion(logits, soft_targets)

        # Combine losses
        total_loss = supervised_loss + (self.lambda_distill * distillation_loss)

        return total_loss
