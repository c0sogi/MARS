import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class LabelSmoothingLoss(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.
    Used for Stage 1 (Teacher) training to handle noisy labels.

    Formula:
    L = (1 - epsilon) * CE(y_true, y_pred) + epsilon * CE(Uniform, y_pred)
    """

    def __init__(self, smoothing=Config.label_smoothing, reduction="mean"):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, target):
        """
        Args:
            logits (torch.Tensor): Predicted logits [batch_size, num_classes]
            target (torch.Tensor): Ground truth indices [batch_size]

        Returns:
            torch.Tensor: Scalar loss (if reduction is 'mean')
        """
        # Number of classes (sequence length)
        c = logits.size(-1)

        # Compute log probabilities
        log_preds = F.log_softmax(logits, dim=-1)

        # Standard Cross Entropy (NLL Loss)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)

        # Loss against Uniform Distribution
        # Sum of log_preds over classes, effectively -sum(log(p))
        # We want mean(-log(p)) which is -sum(log(p)) / C
        loss_uniform = -log_preds.sum(dim=-1)

        if self.reduction == "mean":
            loss_uniform = loss_uniform.mean()

        # Combine components
        # (1 - eps) * Hard_Target_Loss + eps * Uniform_Target_Loss
        # Uniform_Target_Loss is equivalent to (loss_uniform / c)
        return (1.0 - self.smoothing) * nll + self.smoothing * (loss_uniform / c)


class DistillationLoss(nn.Module):
    """
    Combined Loss for Stage 2 (Student) training.
    Combines Cross Entropy (against ground truth) and KL Divergence (against teacher logits).

    Formula:
    L = alpha * CE(y_true, y_pred) + (1 - alpha) * KL(y_teacher, y_pred)
    """

    def __init__(self, alpha=Config.distillation_alpha, temperature=Config.temperature):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature

        # Standard CE for the hard labels
        self.ce_loss = nn.CrossEntropyLoss()

        # KL Divergence for soft labels
        # reduction='batchmean' aligns mathematically with the definition of KL divergence
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits_start,
        student_logits_end,
        teacher_logits_start,
        teacher_logits_end,
        targets_start,
        targets_end,
    ):
        """
        Args:
            student_logits_start: [batch, seq_len]
            student_logits_end:   [batch, seq_len]
            teacher_logits_start: [batch, seq_len]
            teacher_logits_end:   [batch, seq_len]
            targets_start:        [batch] (indices)
            targets_end:          [batch] (indices)
        """

        # 1. Hard Label Loss (Cross Entropy against Ground Truth)
        loss_start_ce = self.ce_loss(student_logits_start, targets_start)
        loss_end_ce = self.ce_loss(student_logits_end, targets_end)
        total_ce = loss_start_ce + loss_end_ce

        # 2. Soft Label Loss (KL Divergence against Teacher)
        T = self.temperature

        # Apply temperature scaling
        # Student: log_softmax (required for KLDiv input)
        student_start_log_prob = F.log_softmax(student_logits_start / T, dim=-1)
        student_end_log_prob = F.log_softmax(student_logits_end / T, dim=-1)

        # Teacher: softmax (targets are probabilities)
        # Detach teacher logits to ensure we don't backprop through the teacher
        with torch.no_grad():
            teacher_start_prob = F.softmax(teacher_logits_start / T, dim=-1)
            teacher_end_prob = F.softmax(teacher_logits_end / T, dim=-1)

        # Compute KL Divergence
        # We multiply by T^2 to scale gradients to be commensurate with CE loss
        loss_start_kl = self.kl_div_loss(student_start_log_prob, teacher_start_prob) * (
            T**2
        )
        loss_end_kl = self.kl_div_loss(student_end_log_prob, teacher_end_prob) * (T**2)
        total_kl = loss_start_kl + loss_end_kl

        # 3. Weighted Combination
        # L = alpha * CE + (1 - alpha) * KL
        return self.alpha * total_ce + (1.0 - self.alpha) * total_kl
