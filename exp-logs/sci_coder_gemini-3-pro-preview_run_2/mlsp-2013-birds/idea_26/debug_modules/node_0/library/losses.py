import torch
import torch.nn as nn
from library.config import CFG


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for multi-label classification.
    Uses pos_weight to handle class imbalance by upweighting the loss contribution
    of positive samples for rare classes.
    """

    def __init__(self, pos_weights=None, device=CFG.device):
        """
        Args:
            pos_weights (list or torch.Tensor, optional): Weights for positive classes.
                                                          If None, uses CFG.pos_weights.
            device (str): Device to place the weights on.
        """
        super(WeightedBCELoss, self).__init__()

        if pos_weights is None:
            pos_weights = CFG.pos_weights

        # Ensure pos_weights is a tensor
        if not isinstance(pos_weights, torch.Tensor):
            self.pos_weights = torch.tensor(pos_weights, dtype=torch.float32)
        else:
            self.pos_weights = pos_weights.clone().detach()

        # Move to device
        self.pos_weights = self.pos_weights.to(device)

        # Initialize BCEWithLogitsLoss
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits from the model (Batch, NumClasses).
            targets (torch.Tensor): Binary ground truth labels (Batch, NumClasses).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        return self.criterion(inputs, targets)


class WeightedDistillationLoss(nn.Module):
    """
    Class-Weighted Knowledge Distillation Loss.
    Combines a supervised loss with a distillation loss against teacher soft targets.
    Crucially, applies class-positive weights to BOTH terms to prevent the student
    from learning the teacher's bias against rare classes.

    Formula:
    L = BCE(Target, Pred, w) + lambda * BCE(Teacher_Probs, Pred, w)
    """

    def __init__(
        self,
        pos_weights=None,
        distillation_lambda=CFG.distillation_lambda,
        device=CFG.device,
    ):
        """
        Args:
            pos_weights (list or torch.Tensor, optional): Weights for positive classes.
            distillation_lambda (float): Weighting factor for the distillation term.
            device (str): Device to place the weights on.
        """
        super(WeightedDistillationLoss, self).__init__()

        self.distillation_lambda = distillation_lambda

        if pos_weights is None:
            pos_weights = CFG.pos_weights

        # Ensure pos_weights is a tensor
        if not isinstance(pos_weights, torch.Tensor):
            self.pos_weights = torch.tensor(pos_weights, dtype=torch.float32)
        else:
            self.pos_weights = pos_weights.clone().detach()

        # Move to device
        self.pos_weights = self.pos_weights.to(device)

        # Initialize criteria
        # Both terms use the same pos_weight to maintain focus on rare classes
        self.supervised_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
        self.distillation_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)

    def forward(self, student_logits, targets, teacher_probs):
        """
        Args:
            student_logits (torch.Tensor): Logits from the student model.
            targets (torch.Tensor): Binary ground truth labels.
            teacher_probs (torch.Tensor): Soft probabilities from the teacher/anchor ensemble.

        Returns:
            torch.Tensor: Combined scalar loss.
        """
        # 1. Supervised Loss (Student vs Ground Truth)
        loss_supervised = self.supervised_criterion(student_logits, targets)

        # 2. Distillation Loss (Student vs Teacher Soft Targets)
        # BCEWithLogitsLoss takes logits as input and probabilities (0-1) as target.
        # teacher_probs are already probabilities (OOFs), so we pass them directly as targets.
        loss_distill = self.distillation_criterion(student_logits, teacher_probs)

        # Combine
        total_loss = loss_supervised + (self.distillation_lambda * loss_distill)

        return total_loss
