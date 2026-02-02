import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridLoss(nn.Module):
    """
    Implements the Decoupled Multi-Task Loss.
    Combines Weighted Cross-Entropy for the main 4-class head with
    Binary Cross-Entropy for the auxiliary 'Rust' and 'Scab' heads.
    """

    def __init__(self, weight=None):
        """
        Args:
            weight (torch.Tensor, optional): Class weights for the main CrossEntropyLoss.
        """
        super(HybridLoss, self).__init__()
        self.weight = weight
        # Initialize CrossEntropyLoss with provided weights
        self.main_loss_fn = nn.CrossEntropyLoss(weight=weight)
        # BCEWithLogitsLoss combines Sigmoid layer and BCELoss in one single class
        self.aux_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Dictionary containing 'main', 'rust', 'scab' logits.
            targets (torch.Tensor): Ground truth tensor of shape (Batch, 4).
                                    [healthy, multiple_diseases, rust, scab]

        Returns:
            torch.Tensor: The scalar loss value.
        """
        # 1. Main Head Loss (Multi-class)
        # targets are one-hot/float, convert to indices for CrossEntropy
        main_target_indices = torch.argmax(targets, dim=1)
        loss_main = self.main_loss_fn(outputs["main"], main_target_indices)

        # 2. Auxiliary Head Losses (Binary)
        # Logic:
        # - Rust Head Target is 1 if class is 'rust' (idx 2) OR 'multiple_diseases' (idx 1)
        # - Scab Head Target is 1 if class is 'scab' (idx 3) OR 'multiple_diseases' (idx 1)

        # Extract columns (assuming order: healthy, multiple, rust, scab)
        is_multiple = targets[:, 1]
        is_rust = targets[:, 2]
        is_scab = targets[:, 3]

        # Create binary targets
        rust_binary_target = (is_multiple + is_rust > 0).float()
        scab_binary_target = (is_multiple + is_scab > 0).float()

        # Compute BCE losses
        # view(-1) ensures shape matches (Batch,)
        loss_rust = self.aux_loss_fn(outputs["rust"].view(-1), rust_binary_target)
        loss_scab = self.aux_loss_fn(outputs["scab"].view(-1), scab_binary_target)

        # Total Loss
        return loss_main + loss_rust + loss_scab


class DistillationLoss(nn.Module):
    """
    Implements Knowledge Distillation Loss.
    Combines the HybridLoss (Student Task Loss) with KL Divergence between
    Student and Teacher logits.
    """

    def __init__(
        self, weight=None, alpha=Config.DISTILLATION_ALPHA, T=Config.TEMPERATURE
    ):
        """
        Args:
            weight (torch.Tensor, optional): Class weights for the HybridLoss.
            alpha (float): Weighting factor for the distillation loss term.
            T (float): Temperature for softening probability distributions.
        """
        super(DistillationLoss, self).__init__()
        self.hybrid_loss = HybridLoss(weight=weight)
        self.alpha = alpha
        self.T = T
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_outputs, teacher_outputs, targets):
        """
        Args:
            student_outputs (dict): Logits from the student model.
            teacher_outputs (dict): Logits from the frozen teacher model.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: The total combined loss.
        """
        # 1. Student Task Loss (Hard Labels + Aux Heads)
        loss_task = self.hybrid_loss(student_outputs, targets)

        # 2. Distillation Loss (Soft Labels)
        # We distill knowledge primarily through the main classification head
        student_logits = student_outputs["main"]

        # Detach teacher logits to ensure gradients don't flow back to teacher
        teacher_logits = teacher_outputs["main"].detach()

        # Compute KL Divergence
        # P (Teacher): Softmax(teacher_logits / T)
        # Q (Student): LogSoftmax(student_logits / T)
        # Note: KLDivLoss expects log_softmax for input (student) and standard probabilities for target (teacher)
        # if log_target=False (default)

        p_teacher = F.softmax(teacher_logits / self.T, dim=1)
        log_q_student = F.log_softmax(student_logits / self.T, dim=1)

        loss_distill = self.kl_div_loss(log_q_student, p_teacher) * (self.T**2)

        # 3. Total Loss
        total_loss = loss_task + self.alpha * loss_distill

        return total_loss
