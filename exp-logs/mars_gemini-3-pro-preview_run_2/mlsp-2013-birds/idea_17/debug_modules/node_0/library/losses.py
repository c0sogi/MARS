import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Custom loss function for Self-Distillation.

    Computes a weighted sum of:
    1. Hard Loss: Weighted BCEWithLogitsLoss between student logits and ground truth labels.
    2. Soft Loss: BCEWithLogitsLoss between student logits and teacher probabilities (soft labels).

    Formula:
        Loss = gamma * Hard_Loss + (1 - gamma) * Soft_Loss
    """

    def __init__(self, gamma=0.5, pos_weight=None):
        """
        Args:
            gamma (float): Mixing coefficient.
                           1.0 = Only Hard Loss (Standard Supervised).
                           0.0 = Only Soft Loss.
                           Default is 0.5.
            pos_weight (torch.Tensor, optional): Weight for positive examples in Hard Loss
                                                 to handle class imbalance.
        """
        super(DistillationLoss, self).__init__()
        self.gamma = gamma

        # Register pos_weight as a buffer so it moves to device with the module
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, student_logits, targets, teacher_probs=None):
        """
        Args:
            student_logits (torch.Tensor): Raw output from the student model (N, C).
            targets (torch.Tensor): Ground truth binary labels (N, C).
            teacher_probs (torch.Tensor, optional): Probabilities from the teacher model (N, C).
                                                    If None, only hard loss is computed.

        Returns:
            torch.Tensor: The computed loss scalar.
        """
        # 1. Compute Hard Loss (Student vs Ground Truth)
        # We use BCEWithLogitsLoss which combines Sigmoid + BCELoss for numerical stability
        hard_loss = F.binary_cross_entropy_with_logits(
            student_logits, targets, pos_weight=self.pos_weight
        )

        # If no teacher probabilities are provided, return hard loss (Standard Supervised)
        if teacher_probs is None:
            return hard_loss

        # 2. Compute Soft Loss (Student vs Teacher)
        # We treat teacher_probs as the target. Since student_logits are logits,
        # we again use BCEWithLogitsLoss.
        soft_loss = F.binary_cross_entropy_with_logits(student_logits, teacher_probs)

        # 3. Combine
        loss = (self.gamma * hard_loss) + ((1 - self.gamma) * soft_loss)

        return loss
