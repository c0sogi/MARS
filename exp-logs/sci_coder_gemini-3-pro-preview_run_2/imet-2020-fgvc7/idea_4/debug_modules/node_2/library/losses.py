import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification.
    Paper: "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)

    This loss function addresses the problem of class imbalance by decoupling the
    focusing levels of the positive and negative samples. It also implements
    probability shifting (clipping) to discard easy negative samples.
    """

    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
    ):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Logits of shape (batch_size, num_classes)
            y (torch.Tensor): Binary targets of shape (batch_size, num_classes)
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Cross Entropy Calculation
        # For positives: - y * log(p)
        # For negatives: - (1-y) * log(1-p) (with shifting applied to 1-p)

        # Positive Loss Component
        # L+ = (1 - p)^gamma_pos * log(p)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_pos = los_pos * (1 - xs_pos).pow(self.gamma_pos)

        # Negative Loss Component
        # L- = (p_shifted)^gamma_neg * log(1 - p_shifted)
        # Here xs_neg represents (1 - p_shifted) effectively in the log term logic
        # but the weight is based on p_shifted.
        # Note: xs_neg is the probability of being negative (shifted).
        # We want loss = -log(xs_neg) * weight.
        # weight = (1 - xs_neg)^gamma_neg.

        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        los_neg = los_neg * (1 - xs_neg).pow(self.gamma_neg)

        # Total Loss
        loss = -los_pos - los_neg

        # Sum over classes, mean over batch
        loss = loss.sum(dim=1)
        return loss.mean()


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss.
    Combines Asymmetric Loss (for hard ground truth targets) and
    Binary Cross Entropy (for soft teacher targets).
    """

    def __init__(self, alpha=Config.DISTILL_ALPHA):
        """
        Args:
            alpha (float): Weight for the soft teacher loss.
                           Loss = alpha * SoftLoss + (1 - alpha) * HardLoss
        """
        super(DistillationLoss, self).__init__()
        self.alpha = alpha

        # Hard Target Loss: Asymmetric Loss to handle imbalance in ground truth
        self.hard_loss_fn = AsymmetricLoss(
            gamma_neg=Config.ASL_GAMMA_NEG,
            gamma_pos=Config.ASL_GAMMA_POS,
            clip=Config.ASL_CLIP,
        )

        # Soft Target Loss: BCEWithLogitsLoss
        # We use BCE because the teacher probabilities are already calibrated/softened
        # and we want the student logits to match that distribution.
        self.soft_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, student_logits, teacher_probs, hard_targets):
        """
        Args:
            student_logits (torch.Tensor): Raw logits from student model (N, C)
            teacher_probs (torch.Tensor): Probabilities from teacher ensemble (N, C)
            hard_targets (torch.Tensor): Binary ground truth labels (N, C)
        """
        # Calculate Hard Loss (Student vs Ground Truth)
        loss_hard = self.hard_loss_fn(student_logits, hard_targets)

        # Calculate Soft Loss (Student vs Teacher)
        # BCEWithLogitsLoss takes logits as input and applies Sigmoid internally
        loss_soft = self.soft_loss_fn(student_logits, teacher_probs)

        # Weighted Combination
        loss = (self.alpha * loss_soft) + ((1 - self.alpha) * loss_hard)

        return loss
