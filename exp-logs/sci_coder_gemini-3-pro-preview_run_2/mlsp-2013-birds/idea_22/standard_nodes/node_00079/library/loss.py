import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    Binary Cross Entropy Loss with positive class weighting.
    Used for training the stable anchors (ResNet18, EfficientNet) to handle class imbalance.
    """

    def __init__(self, pos_weights=None):
        """
        Args:
            pos_weights (torch.Tensor, list, or numpy array, optional):
                Weights for positive classes. Shape: (num_classes,).
                If provided, these are registered as a buffer to handle device placement.
        """
        super(WeightedBCELoss, self).__init__()
        if pos_weights is not None:
            if not isinstance(pos_weights, torch.Tensor):
                pos_weights = torch.tensor(pos_weights)
            # Register as buffer so it moves to device when loss_fn.to(device) is called
            self.register_buffer("pos_weights", pos_weights)
        else:
            self.pos_weights = None

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid). Shape (Batch, Num_Classes).
            targets (torch.Tensor): Binary ground truth labels. Shape (Batch, Num_Classes).

        Returns:
            torch.Tensor: The calculated loss.
        """
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weights
        )


class AnchorDistillationLoss(nn.Module):
    """
    Composite loss function for the DenseNet student model.
    Combines a weighted classification loss against hard targets with a
    distillation loss against soft targets provided by the anchor models.
    """

    def __init__(self, pos_weights=None, distillation_lambda=1.0):
        """
        Args:
            pos_weights (torch.Tensor, optional): Weights for positive classes (for the hard loss term).
            distillation_lambda (float): Weighting factor for the distillation term (KL Divergence proxy).
        """
        super(AnchorDistillationLoss, self).__init__()
        self.hard_loss_fn = WeightedBCELoss(pos_weights)
        self.distillation_lambda = distillation_lambda

    def forward(self, student_logits, hard_targets, soft_targets):
        """
        Args:
            student_logits (torch.Tensor): Raw outputs from the student (DenseNet). Shape (B, C).
            hard_targets (torch.Tensor): Binary ground truth labels. Shape (B, C).
            soft_targets (torch.Tensor): Soft probabilities from the anchor ensemble. Shape (B, C).

        Returns:
            torch.Tensor: The combined weighted loss.
        """
        # 1. Hard Loss: Weighted BCE against ground truth labels
        # This ensures the model learns the primary classification task, respecting class imbalance.
        loss_hard = self.hard_loss_fn(student_logits, hard_targets)

        # 2. Distillation Loss: Student vs Anchor Soft Targets
        # Minimizing Binary Cross Entropy between the student logits and the soft teacher probabilities
        # is mathematically equivalent to minimizing the Binary KL Divergence (up to a constant
        # entropy term of the teacher). We use BCEWithLogitsLoss for numerical stability.
        # Note: No pos_weight is used here to faithfully match the teacher's distribution.
        loss_distill = F.binary_cross_entropy_with_logits(
            student_logits, soft_targets, reduction="mean"
        )

        # Composite Loss
        return loss_hard + (self.distillation_lambda * loss_distill)
