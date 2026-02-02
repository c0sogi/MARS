import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification.

    This loss function addresses class imbalance by down-weighting easy negative examples.
    It operates on logits and applies asymmetric focusing parameters and probability clipping.

    Reference: "Asymmetric Loss For Multi-Label Classification" (Ben-Baruch et al.)
    """

    def __init__(
        self,
        gamma_neg=Config.ASL_GAMMA_NEG,
        gamma_pos=Config.ASL_GAMMA_POS,
        clip=Config.ASL_CLIP,
        eps=1e-8,
    ):
        """
        Args:
            gamma_neg (float): Focusing parameter for negative examples. Higher values down-weight easy negatives.
            gamma_pos (float): Focusing parameter for positive examples.
            clip (float): Probability margin for shifting/clipping negative probabilities.
            eps (float): Epsilon for numerical stability in logarithms.
        """
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Logits from the model (batch_size, num_classes).
            y (torch.Tensor): Multi-hot ground truth targets (batch_size, num_classes).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculate probabilities from logits
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        # Shifts the negative probabilities to filter out very easy negatives
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Cross Entropy Calculation
        # We use clamp(min=eps) to avoid log(0)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing Weights
        # For positives (y=1), weight is (1-p)^gamma_pos
        coeff_pos = (1 - xs_pos).pow(self.gamma_pos)

        # For negatives (y=0), weight is (1 - p_neg)^gamma_neg
        # where p_neg is the probability of the negative class (shifted)
        coeff_neg = (1 - xs_neg).pow(self.gamma_neg)

        # Combine terms
        loss = -(coeff_pos * los_pos + coeff_neg * los_neg)

        # Sum over classes, mean over batch
        loss = loss.sum(dim=1).mean()

        return loss


class DistillationLoss(nn.Module):
    """
    Composite loss function for Knowledge Distillation in Multi-Label Classification.

    Combines:
    1. Asymmetric Loss computed against hard ground truth labels.
    2. KL Divergence (approximated via Soft BCE) computed against soft teacher probabilities.
    """

    def __init__(
        self,
        alpha=Config.DISTILLATION_ALPHA,
        temp=Config.DISTILLATION_TEMP,
        gamma_neg=Config.ASL_GAMMA_NEG,
        gamma_pos=Config.ASL_GAMMA_POS,
        clip=Config.ASL_CLIP,
    ):
        """
        Args:
            alpha (float): Weighting factor for the soft distillation loss (0.0 to 1.0).
            temp (float): Temperature for scaling logits in distillation.
            gamma_neg, gamma_pos, clip: Parameters for the internal AsymmetricLoss.
        """
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temp = temp

        # Hard label loss component
        self.asl_loss = AsymmetricLoss(
            gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip
        )

        # Soft label loss component
        # Using BCEWithLogitsLoss is standard for multi-label distillation
        # It is numerically stable and equivalent to minimizing binary KL divergence
        self.bce_soft = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, student_logits, teacher_logits, targets):
        """
        Args:
            student_logits (torch.Tensor): Logits from the student model.
            teacher_logits (torch.Tensor): Logits from the teacher model (or ensemble).
            targets (torch.Tensor): Hard ground truth labels.

        Returns:
            torch.Tensor: Weighted combined loss.
        """
        # 1. Hard Label Loss (ASL)
        loss_hard = self.asl_loss(student_logits, targets)

        # 2. Soft Label Loss (Distillation)
        # Apply temperature scaling
        student_soft = student_logits / self.temp
        teacher_soft = teacher_logits / self.temp

        # Convert teacher logits to probabilities (soft targets)
        teacher_probs = torch.sigmoid(teacher_soft)

        # Calculate BCE with soft targets
        # We sum over classes and mean over batch to match ASL reduction scale
        loss_soft_bce = self.bce_soft(student_soft, teacher_probs)
        loss_soft = loss_soft_bce.sum(dim=1).mean()

        # Scale by T^2 to maintain gradient magnitude relative to hard loss
        # This is a standard practice in knowledge distillation
        loss_soft = loss_soft * (self.temp**2)

        # Weighted combination
        loss = (1 - self.alpha) * loss_hard + self.alpha * loss_soft

        return loss
