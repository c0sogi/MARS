import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Custom Loss function for Knowledge Distillation.
    Combines Weighted Cross Entropy Loss (Hard Target) and KL Divergence Loss (Soft Target).

    Formula:
        Loss = alpha * WeightedCE(y_true, y_student) + (1 - alpha) * KLDiv(sigma(z_teacher/T), sigma(z_student/T))
    """

    def __init__(self, class_weights=None, alpha=0.5, temperature=4.0):
        """
        Args:
            class_weights (torch.Tensor, optional): Weights for each class to handle imbalance.
                                                  Passed to CrossEntropyLoss.
            alpha (float): Weighting factor for the Hard Loss (CrossEntropy).
                           Soft Loss (KLDiv) will be weighted by (1 - alpha).
            temperature (float): Temperature parameter T to soften probability distributions
                                 for distillation.
        """
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature

        # Hard Loss: Weighted Cross Entropy
        # Handles class imbalance by penalizing minority classes more heavily.
        # If class_weights is provided, it must be a Tensor.
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

        # Soft Loss: KL Divergence
        # reduction='batchmean' is used to align with the mathematical definition of KL Divergence
        # averaged over the batch size.
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, labels, teacher_logits=None):
        """
        Computes the combined distillation loss.

        Args:
            student_logits (torch.Tensor): Raw output logits from the student model. Shape: (B, C)
            labels (torch.Tensor): Ground truth class indices. Shape: (B,)
            teacher_logits (torch.Tensor, optional): Raw output logits from the teacher model.
                                                     Shape: (B, C). If None, only Hard Loss is computed.

        Returns:
            torch.Tensor: The calculated loss scalar.
        """
        # 1. Calculate Hard Loss (Weighted Cross Entropy)
        # This ensures the model learns the correct ground truth labels, prioritizing minority classes.
        hard_loss = self.ce_loss(student_logits, labels)

        # 2. Check if Teacher Logits are provided (Distillation Mode)
        if teacher_logits is None:
            # If no teacher logits, we are likely in a standard training phase or validation
            return hard_loss

        # 3. Calculate Soft Loss (KL Divergence)
        # Distillation involves matching the "dark knowledge" (soft probabilities) of the teacher.

        # Apply Temperature Scaling
        # Student output must be Log-Softmax for PyTorch's KLDivLoss
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)

        # Teacher output must be Softmax (probabilities) for PyTorch's KLDivLoss
        # We detach teacher logits to ensure gradients don't flow back to the teacher (though usually teacher is frozen)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)

        # Compute KL Divergence
        soft_loss = self.kl_div_loss(student_log_probs, teacher_probs)

        # 4. Combine Losses
        # Formula: L = alpha * HardLoss + (1 - alpha) * SoftLoss
        loss = (self.alpha * hard_loss) + ((1.0 - self.alpha) * soft_loss)

        return loss
