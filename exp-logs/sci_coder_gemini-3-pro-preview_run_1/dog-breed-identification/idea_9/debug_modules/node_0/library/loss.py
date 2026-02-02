import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DistillationLoss(nn.Module):
    """
    Custom Loss function for Knowledge Distillation.
    Combines Cross-Entropy Loss (Hard Labels) and KL Divergence Loss (Soft Targets).

    Formula:
    Loss = (1 - alpha) * CE(y_true, y_pred) + alpha * T^2 * KL(sigma(z_teacher/T), sigma(z_student/T))
    """

    def __init__(self, config: Config):
        """
        Args:
            config (Config): Configuration object containing 'distillation_alpha'
                             and 'distillation_temp'.
        """
        super(DistillationLoss, self).__init__()
        self.alpha = config.distillation_alpha
        self.temperature = config.distillation_temp

        # Standard Cross Entropy for hard labels
        self.ce_loss = nn.CrossEntropyLoss()

        # KL Divergence for soft targets
        # reduction='batchmean' aligns with the mathematical definition of KL divergence
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Computes the distilled loss.

        Args:
            student_logits (torch.Tensor): Logits from the student model (Batch, Num_Classes).
            labels (torch.Tensor): Ground truth hard labels (Batch).
            teacher_logits (torch.Tensor, optional): Logits from the teacher model (Batch, Num_Classes).

        Returns:
            torch.Tensor: The computed loss value.
        """
        # 1. Compute Hard Loss (Cross Entropy)
        hard_loss = self.ce_loss(student_logits, labels)

        # If no teacher logits are provided (e.g., during Stage 1 or validation), return hard loss
        if teacher_logits is None:
            return hard_loss

        # 2. Compute Soft Loss (KL Divergence)
        # Apply temperature scaling
        # Student: Log-Softmax (required by nn.KLDivLoss input)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)

        # Teacher: Softmax (required by nn.KLDivLoss target)
        # We detach teacher logits to ensure no gradients flow back to the teacher (though usually teacher is frozen/offline)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)

        distillation_loss = self.kl_div_loss(student_log_probs, teacher_probs)

        # 3. Combine Losses
        # Scale distillation loss by T^2 to match the scale of gradients produced by hard labels
        loss = (1 - self.alpha) * hard_loss + (
            self.alpha * (self.temperature**2) * distillation_loss
        )

        return loss
